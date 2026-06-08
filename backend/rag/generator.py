"""Answer generation with local phi3 through Ollama."""

from operator import itemgetter

from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from backend.core.config import get_settings

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    "about",
    "tell",
    "explain",
    "give",
    "me",
}


def _format_context(documents: list[Document], max_chars: int = 9000) -> str:
    parts: list[str] = []
    used = 0
    for index, document in enumerate(documents, start=1):
        filename = str(document.metadata.get("filename", "document.pdf"))
        page = int(document.metadata.get("page_number", 0))
        label = f"Source {index}: {filename} (Page {page})"
        text = " ".join(document.page_content.split())
        remaining = max_chars - used - len(label) - 16
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rsplit(" ", 1)[0]
        piece = f"{label}\nExcerpt:\n{text}"
        parts.append(piece)
        used += len(piece)
    return "\n\n".join(parts)


def _format_sources(documents: list[Document]) -> str:
    unique: list[tuple[str, int]] = []
    for document in documents:
        item = (str(document.metadata.get("filename", "document.pdf")), int(document.metadata.get("page_number", 0)))
        if item not in unique:
            unique.append(item)
    lines = ["Sources:"]
    lines.extend(f"* {filename} (Page {page})" for filename, page in unique)
    return "\n".join(lines)


def _strip_model_sources(answer: str) -> str:
    """Remove model-written source blocks so citations stay deterministic."""

    marker = "\nSources:"
    if marker in answer:
        return answer.split(marker, 1)[0].strip()
    if answer.strip().startswith("Sources:"):
        return ""
    return answer.strip()


def _content_terms(text: str) -> set[str]:
    """Extract simple content terms for a conservative evidence check."""

    cleaned = "".join(char.lower() if char.isalnum() else " " for char in text)
    return {term for term in cleaned.split() if len(term) >= 3 and term not in _STOPWORDS}


def _has_minimum_evidence(question: str, documents: list[Document]) -> bool:
    """Return whether retrieved chunks share enough concrete terms with the question."""

    question_terms = _content_terms(question)
    if not question_terms:
        return True
    context_terms = _content_terms(" ".join(document.page_content for document in documents))
    overlap = question_terms.intersection(context_terms)
    required = 1 if len(question_terms) <= 2 else 2
    return len(overlap) >= required


def generate_answer(question: str, documents: list[Document], history: list[dict[str, str]]) -> str:
    """Generate an answer using ChatOllama(model='phi3:mini') and LCEL."""

    if not documents:
        return "I could not find relevant information in your uploaded PDFs.\n\nSources:\n* No sources found"
    if not _has_minimum_evidence(question, documents):
        return (
            "The uploaded PDFs do not contain enough information to answer that accurately.\n\n"
            f"{_format_sources(documents)}"
        )

    settings = get_settings()
    llm = ChatOllama(model=settings.generation_model, base_url=settings.ollama_base_url, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an offline PDF question-answering assistant. "
                "Use only the PDF excerpts in Context as factual evidence. "
                "Do not use outside knowledge. Do not guess. Do not fill gaps from memory. "
                "If the Context does not directly support the answer, say: "
                "'The uploaded PDFs do not contain enough information to answer that accurately.' "
                "When answering, keep claims narrow and match the wording of the excerpts. "
                "Prefer short answers over broad explanations. "
                "Do not write your own Sources section.",
            ),
            (
                "human",
                "Context from retrieved PDF chunks:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer from the Context only. If support is weak or missing, say the PDFs do not contain enough "
                "information to answer accurately.",
            ),
        ]
    )
    formatted_history = "\n".join(f"{item['role']}: {item['message']}" for item in history[-12:])
    chain = (
        {
            "question": itemgetter("question") | RunnablePassthrough(),
            "context": itemgetter("documents") | RunnablePassthrough() | _format_context,
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = _strip_model_sources(chain.invoke({"question": question, "documents": documents, "history": formatted_history}))
    if not answer:
        answer = "The uploaded PDFs do not contain enough information to answer that accurately."
    return f"{answer}\n\n{_format_sources(documents)}"


def generate_general_answer(question: str, history: list[dict[str, str]]) -> str:
    """Generate a general local chatbot answer using phi3:mini without document retrieval."""

    settings = get_settings()
    llm = ChatOllama(model=settings.generation_model, base_url=settings.ollama_base_url, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful local chatbot running fully offline. "
                "Do not claim access to cloud services or live internet information.",
            ),
            ("human", "Chat history:\n{history}\n\nUser message: {question}"),
        ]
    )
    formatted_history = "\n".join(f"{item['role']}: {item['message']}" for item in history[-12:])
    chain = (
        {
            "question": itemgetter("question") | RunnablePassthrough(),
            "history": itemgetter("history") | RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke({"question": question, "history": formatted_history}).strip()
