from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from litellm import completion
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from tenacity import retry, wait_exponential
from openai import OpenAI

load_dotenv(override=True)

DB_PATH = str(Path(__file__).parent.parent / "vector_db")
MODEL = "gpt-4.1-mini"
RERANK_MODEL = "openai/gpt-4.1-nano"  # cheaper model for reranking
EMBEDDINGS_MODEL = "text-embedding-3-large"
RETRIEVAL_K = 20
FINAL_K = 5

wait = wait_exponential(multiplier=1, min=10, max=240)

openai_client = OpenAI()
embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
llm = ChatOpenAI(model=MODEL, temperature=0, streaming=True)

vector_store = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

SYSTEM_PROMPT = """
You are a knowledgeable and friendly professor representing our university.
You are going to explain the questions that the people are asking about subject lectures etc.
By the way you currently have the Calculus, Computer Techniques and Architecture, Introduction to Java programming, Japanese culture and Management.
Be polite, friendly, and if you do not know something, say it.
If the given context is relevant, use the given context to answer questions.

Context:
{context}
"""


# ── Models ────────────────────────────────────────────────────────────────────

class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

@retry(wait=wait)
def rewrite_query(question: str, history: list[dict] = []) -> str:
    """Rewrite the user's question into a focused knowledge base query."""
    message = f"""
You are a university professor's assistant answering student questions.
You are about to search a Knowledge Base of lecture materials to answer a student's question.

This is the conversation history so far:
{history}

And this is the student's current question:
{question}

Respond ONLY with a short, refined query most likely to surface the relevant lecture content.
Focus on specific concepts, topics, or terms mentioned. No preamble, just the query.
"""
    response = completion(model=RERANK_MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


@retry(wait=wait)
def rerank(question: str, docs: list[Document]) -> list[Document]:
    """Re-rank retrieved documents by relevance to the question."""
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text retrieved from a knowledge base.
Rank order the provided chunks by relevance to the question, most relevant first.
Reply only with the JSON list of ranked chunk ids. Include ALL chunk ids you are provided with.
"""
    user_prompt = f"Question:\n\n{question}\n\nRank all chunks from most to least relevant. Include all chunk ids.\n\n"
    user_prompt += "Chunks:\n\n"
    for i, doc in enumerate(docs):
        user_prompt += f"# CHUNK ID: {i + 1}:\n\n{doc.page_content}\n\n"
    user_prompt += "Reply only with the ranked chunk id list, nothing else."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = completion(model=RERANK_MODEL, messages=messages, response_format=RankOrder)
    order = RankOrder.model_validate_json(response.choices[0].message.content).order
    return [docs[i - 1] for i in order if 1 <= i <= len(docs)]


def merge_docs(primary: list[Document], secondary: list[Document]) -> list[Document]:
    """Merge two document lists, deduplicating by page content."""
    merged = primary[:]
    seen = {doc.page_content for doc in primary}
    for doc in secondary:
        if doc.page_content not in seen:
            merged.append(doc)
            seen.add(doc.page_content)
    return merged


def fetch_docs_unranked(query: str) -> list[Document]:
    """Embed a query and retrieve the top-K docs from the vector store."""
    return vector_store.similarity_search(query, k=RETRIEVAL_K)


def fetch_context(question: str, history: list[dict] = []) -> list[Document]:
    """
    Full retrieval pipeline:
      1. Fetch docs for the original question
      2. Rewrite the query and fetch again
      3. Merge + deduplicate
      4. Rerank and return top FINAL_K
    """
    rewritten = rewrite_query(question, history)
    docs_original = fetch_docs_unranked(question)
    docs_rewritten = fetch_docs_unranked(rewritten)
    merged = merge_docs(docs_original, docs_rewritten)
    reranked = rerank(question, merged)
    return reranked[:FINAL_K]


def combined_query(question: str, history: list[dict]) -> str:
    """Build a combined query string from history + current question for richer retrieval."""
    past = "\n".join(m["content"] for m in history if m["role"] == "user")
    return (past + "\n" + question).strip()


# ── Main entry point ──────────────────────────────────────────────────────────

def stream_answer_question(question: str, history: list[dict]):
    """
    Stream an answer using the full RAG pipeline.
    Yields (partial_answer, docs) tuples.
    """
    combined = combined_query(question, history)
    docs = fetch_context(combined, history)

    context = "\n\n".join(
        f"Extract from {doc.metadata.get('source', 'lecture material')}:\n{doc.page_content}"
        for doc in docs
    )

    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(question))

    partial = ""
    for chunk in llm.stream(messages):
        partial += chunk.content
        yield partial, docs