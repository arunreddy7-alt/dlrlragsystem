"""Answer generation with local phi3 through Ollama."""

from operator import itemgetter

from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from backend.core.config import get_settings


def _format_context(documents: list[Document], max_chars: int = 9000) -> str:
    """Format retrieved chunks into a context string."""

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
    """Create deterministic source citations."""

    unique: list[tuple[str, int]] = []

    for document in documents:
        item = (
            str(document.metadata.get("filename", "document.pdf")),
            int(document.metadata.get("page_number", 0)),
        )

        if item not in unique:
            unique.append(item)

    lines = ["Sources:"]

    lines.extend(
        f"* {filename} (Page {page})"
        for filename, page in unique
    )

    return "\n".join(lines)


def _strip_model_sources(answer: str) -> str:
    """Remove model-generated source blocks."""

    marker = "\nSources:"

    if marker in answer:
        return answer.split(marker, 1)[0].strip()

    if answer.strip().startswith("Sources:"):
        return ""

    return answer.strip()


def generate_answer(
    question: str,
    documents: list[Document],
    history: list[dict[str, str]],
) -> str:
    """
    Generate answer using retrieved PDF chunks.
    """

    if not documents:
        return (
            "I could not find relevant information in your uploaded PDFs.\n\n"
            "Sources:\n* No sources found"
        )

    settings = get_settings()

    llm = ChatOllama(
        model=settings.generation_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an offline PDF question-answering assistant.

Use ONLY the provided PDF context.

Rules:
- Answer using the retrieved PDF content.
- Do not use outside knowledge.
- Do not invent information.
- If the answer is partially available, provide the available information.
- If the answer truly does not exist in the context, respond exactly:
"The uploaded PDFs do not contain enough information to answer that accurately."
- Do not generate a Sources section.
- Give concise but complete answers.
                """,
            ),
            (
                "human",
                """
Context:
{context}

Question:
{question}

Answer:
                """,
            ),
        ]
    )

    # Debug retrieved chunks
    print("\n========== RETRIEVED CHUNKS ==========")

    for index, doc in enumerate(documents, start=1):
        print(f"\nChunk {index}")
        print(doc.page_content[:500])

    print("\n======================================\n")

    chain = (
        {
            "question": itemgetter("question") | RunnablePassthrough(),
            "context": itemgetter("documents")
            | RunnablePassthrough()
            | _format_context,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "question": question,
            "documents": documents,
            "history": history,
        }
    )

    answer = _strip_model_sources(answer)

    if not answer:
        answer = (
            "The uploaded PDFs do not contain enough information "
            "to answer that accurately."
        )

    return f"{answer}\n\n{_format_sources(documents)}"


def generate_general_answer(
    question: str,
    history: list[dict[str, str]],
) -> str:
    """General chatbot mode without retrieval."""

    settings = get_settings()

    llm = ChatOllama(
        model=settings.generation_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful offline AI assistant.

You do not have internet access.
You do not have access to cloud services.
Answer as clearly as possible.
                """,
            ),
            (
                "human",
                """
Chat History:
{history}

User Message:
{question}
                """,
            ),
        ]
    )

    formatted_history = "\n".join(
        f"{item['role']}: {item['message']}"
        for item in history[-12:]
    )

    chain = (
        {
            "question": itemgetter("question")
            | RunnablePassthrough(),
            "history": itemgetter("history")
            | RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(
        {
            "question": question,
            "history": formatted_history,
        }
    ).strip()