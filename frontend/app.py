"""Streamlit frontend for the offline RAG assistant."""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4

import streamlit as st

st.set_page_config(page_title="DLRL RAG Assistant", page_icon="PDF", layout="wide")

LOGO_PATH = Path(__file__).parent / "assets" / "dlrl-logo.png"


def initialize_state() -> None:
    """Initialize Streamlit session state."""

    st.session_state.setdefault("api_url", os.environ.get("BACKEND_API_URL", "http://HOST_MACHINE_IP:8000"))
    st.session_state.setdefault("token", "")
    st.session_state.setdefault("username", "")


def render_header() -> None:
    """Render the DLRL-branded app header."""

    logo_col, title_col = st.columns([1, 6], vertical_alignment="center")
    with logo_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=82)
    with title_col:
        st.markdown(
            """
            <div style="display:flex;align-items:center;height:82px;">
              <h1 style="margin:0;font-size:2.25rem;font-weight:700;">RAG System</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )


def api_request(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    file_payload: tuple[str, bytes, str] | None = None,
) -> Any:
    """Call the FastAPI backend using only the standard library."""

    base_url = st.session_state.api_url.rstrip("/")
    if "HOST_MACHINE_IP" in base_url:
        raise ValueError("Set the backend API URL in Settings before connecting.")
    url = f"{base_url}{path}"
    headers: dict[str, str] = {}
    data: bytes | None = None
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    if file_payload is not None:
        filename, content, content_type = file_payload
        boundary = f"offline-rag-{uuid4().hex}"
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        body = [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            content,
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ]
        data = b"".join(body)
    elif payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            content = response.read()
            if response.status == 204:
                return None
            return json.loads(content.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8")
        try:
            detail = json.loads(message).get("detail", message)
        except json.JSONDecodeError:
            detail = message
        raise ValueError(str(detail)) from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Cannot reach backend: {exc.reason}") from exc
    except (ConnectionResetError, TimeoutError, socket.timeout, OSError) as exc:
        raise ValueError(
            "The backend connection was closed before a response was received. "
            "Confirm the Backend API URL points to FastAPI on port 8000, "
            "then check the backend terminal for the error that occurred while handling this request."
        ) from exc


def login_page() -> None:
    """Render login and registration controls."""

    login_tab, register_tab = st.tabs(["Login", "Registration"])
    with login_tab:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", type="primary"):
            try:
                data = api_request("POST", "/login", {"username": username, "password": password})
                st.session_state.token = data["access_token"]
                st.session_state.username = username
                st.success("Logged in")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with register_tab:
        username = st.text_input("Username", key="register_username")
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        if st.button("Register", type="primary"):
            try:
                data = api_request("POST", "/register", {"username": username, "email": email, "password": password})
                st.session_state.token = data["access_token"]
                st.session_state.username = username
                st.success("Registered and logged in")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))


def settings_page() -> None:
    """Render backend API URL settings."""

    st.subheader("Settings")
    api_url = st.text_input("Backend API URL", value=st.session_state.api_url)
    if st.button("Save Settings"):
        st.session_state.api_url = api_url.strip()
        st.success("Settings saved")
    st.caption("Use the LAN address of the machine running FastAPI, for example http://192.168.1.100:8000.")


def dashboard_page() -> None:
    """Render PDF upload, list, and delete controls."""

    st.subheader("Dashboard")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded is not None and st.button("Upload PDF", type="primary"):
        try:
            api_request("POST", "/upload", file_payload=(uploaded.name, uploaded.read(), "application/pdf"))
            st.success("PDF uploaded and indexed")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    try:
        documents = api_request("GET", "/documents")
    except ValueError as exc:
        st.error(str(exc))
        documents = []

    if not documents:
        st.info("No PDFs uploaded.")
        return
    for document in documents:
        col_name, col_time, col_action = st.columns([4, 3, 1])
        col_name.write(document["filename"])
        col_time.write(document["upload_time"])
        if col_action.button("Delete", key=f"delete_{document['id']}"):
            try:
                api_request("DELETE", f"/documents/{document['id']}")
                st.success("PDF deleted and indexes rebuilt")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))


def clear_data_page() -> None:
    """Render local data clearing controls."""

    st.subheader("Clear Data")
    st.warning("This removes your uploaded PDFs, local indexes, and chat history. Your account remains active.")
    confirm = st.checkbox("I want to clear my local data")
    if st.button("Clear Data", disabled=not confirm):
        try:
            api_request("DELETE", "/data/clear")
            st.success("Local data cleared")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))


def render_chat(endpoint: str, input_label: str, spinner_text: str, show_history: bool = True) -> None:
    """Render a chat box backed by a backend endpoint."""

    if show_history:
        try:
            history = api_request("GET", "/chat/history")
        except ValueError as exc:
            st.error(str(exc))
            history = []

        for item in history:
            with st.chat_message(item["role"]):
                st.markdown(item["message"])

    prompt = st.chat_input(input_label)
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner(spinner_text):
                try:
                    result = api_request("POST", endpoint, {"message": prompt})
                    st.markdown(result["answer"])
                except ValueError as exc:
                    st.error(str(exc))


def chat_page(documents: list[dict[str, Any]]) -> None:
    """Render document chat after upload and a separate general chatbot."""

    st.subheader("Chat")
    pdf_tab, general_tab = st.tabs(["PDF Chat", "General Chatbot"])
    with pdf_tab:
        if not documents:
            st.info("Upload a PDF in the Dashboard before using PDF Chat.")
        else:
            render_chat("/chat", "Ask a question about your uploaded PDFs", "Retrieving sources and generating an answer...")
    with general_tab:
        render_chat("/chat/general", "Chat locally with phi3:mini", "Generating a local response...")


def load_documents() -> list[dict[str, Any]]:
    """Load document metadata for UI routing."""

    try:
        documents = api_request("GET", "/documents")
    except ValueError as exc:
        st.error(str(exc))
        return []
    return documents


def app() -> None:
    """Render the Streamlit app."""

    initialize_state()
    render_header()
    settings_page()
    if not st.session_state.token:
        login_page()
        return
    st.sidebar.write(f"Signed in as {st.session_state.username or 'local user'}")
    if st.sidebar.button("Logout"):
        st.session_state.token = ""
        st.session_state.username = ""
        st.rerun()
    documents = load_documents()
    if documents:
        dashboard_tab, chat_tab, clear_tab = st.tabs(["Dashboard", "Chat", "Clear Data"])
    else:
        dashboard_tab, general_tab, clear_tab = st.tabs(["Dashboard", "General Chatbot", "Clear Data"])
    with dashboard_tab:
        dashboard_page()
    if documents:
        with chat_tab:
            chat_page(documents)
    else:
        with general_tab:
            st.subheader("General Chatbot")
            render_chat("/chat/general", "Chat locally with phi3:mini", "Generating a local response...")
    with clear_tab:
        clear_data_page()


if __name__ == "__main__":
    app()
