import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredPowerPointLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from litellm import completion
from tenacity import retry, wait_exponential
from multiprocessing import Pool
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)

DATA_PATH = str(Path(__file__).parent.parent / "subject_documents")
DB_PATH = str(Path(__file__).parent.parent / "vector_db")

CHUNKING_MODEL = "openai/gpt-4.1-nano"   # cheap — runs once per document
EMBEDDINGS_MODEL = "text-embedding-3-large"
AVERAGE_CHUNK_SIZE = 150                  # target words per chunk
WORKERS = 3                               # reduce to 1 if you hit rate limits

wait = wait_exponential(multiplier=1, min=10, max=240)
EMBEDDINGS = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)


# ── Pydantic models ───────────────────────────────────────────────────────────

class Chunk(BaseModel):
    headline: str = Field(
        description="A brief heading for this chunk, a few words, most likely to match a student's query"
    )
    summary: str = Field(
        description="A few sentences summarising this chunk's content to answer common student questions"
    )
    original_text: str = Field(
        description="The original text of this chunk, exactly as it appears in the document, unchanged"
    )

    def as_document(self, source: str, doc_type: str) -> Document:
        """Pack headline + summary + original text into a single retrievable Document."""
        return Document(
            page_content=self.headline + "\n\n" + self.summary + "\n\n" + self.original_text,
            metadata={"source": source, "doc_type": doc_type},
        )


class Chunks(BaseModel):
    chunks: list[Chunk]


# ── Document loading ──────────────────────────────────────────────────────────

def load_documents() -> list[dict]:
    """
    Load all PDFs and PPTX files from subject_documents subfolders.
    Returns raw dicts with keys: type, source, text.
    """
    raw_docs = []
    for folder in glob.glob(str(Path(DATA_PATH) / "*")):
        doc_type = os.path.basename(folder)

        loader_pdf = DirectoryLoader(path=folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
        loader_pptx = DirectoryLoader(path=folder, glob="**/*.pptx", loader_cls=UnstructuredPowerPointLoader)

        for lc_doc in loader_pdf.load() + loader_pptx.load():
            raw_docs.append({
                "type": doc_type,
                "source": lc_doc.metadata.get("source", "unknown"),
                "text": lc_doc.page_content,
            })

    print(f"Loaded {len(raw_docs)} pages/slides")
    return raw_docs


# ── LLM chunking ──────────────────────────────────────────────────────────────

def make_prompt(document: dict) -> str:
    how_many = max(1, len(document["text"].split()) // AVERAGE_CHUNK_SIZE)
    return f"""
You take a document and split it into overlapping semantic chunks for a university Knowledge Base.

The document is from the subject folder: {document["type"]}
The document was retrieved from: {document["source"]}

A professor's AI assistant will use these chunks to answer student questions about lectures, topics, and concepts.
Divide the document as you see fit, ensuring the ENTIRE document is represented across the chunks — don't leave anything out.
Aim for roughly {how_many} chunks, but use more or fewer as appropriate.
Include ~25% overlap between adjacent chunks (about 50 words) so that context is preserved across chunk boundaries.

For each chunk provide:
- headline: a short phrase a student might search for
- summary: 2–3 sentences answering the most likely student questions about this section
- original_text: the exact original text of this chunk, word for word, unchanged

Here is the document:

{document["text"]}

Respond with the chunks.
"""


@retry(wait=wait)
def process_document(document: dict) -> list[Document]:
    """Send one page/slide to the LLM for semantic chunking."""
    messages = [{"role": "user", "content": make_prompt(document)}]
    response = completion(model=CHUNKING_MODEL, messages=messages, response_format=Chunks)
    doc_chunks = Chunks.model_validate_json(response.choices[0].message.content).chunks
    return [chunk.as_document(document["source"], document["type"]) for chunk in doc_chunks]


def create_chunks(documents: list[dict]) -> list[Document]:
    """
    Run LLM chunking in parallel across all documents.
    Set WORKERS=1 if you hit rate limits.
    """
    all_chunks = []
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            all_chunks.extend(result)
    print(f"Created {len(all_chunks)} semantic chunks")
    return all_chunks


# ── Vector store ──────────────────────────────────────────────────────────────

def store_to_database(chunks: list[Document]):
    """
    Upsert chunks into ChromaDB, skipping any that are already stored.
    IDs are derived from source path + chunk index so re-runs are idempotent.
    """
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=EMBEDDINGS)

    existing_ids = set(vector_store.get(include=[])["ids"])
    print(f"Existing documents in DB: {len(existing_ids)}")

    # Assign stable IDs: "<source>:<sequential index>"
    new_chunks, new_ids = [], []
    source_counters: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        idx = source_counters.get(source, 0)
        chunk_id = f"{source}:{idx}"
        source_counters[source] = idx + 1

        if chunk_id not in existing_ids:
            chunk.metadata["id"] = chunk_id
            new_chunks.append(chunk)
            new_ids.append(chunk_id)

    if new_chunks:
        print(f"Adding {len(new_chunks)} new chunks to DB")
        vector_store.add_documents(documents=new_chunks, ids=new_ids)
    else:
        print("No new documents to add")


if __name__ == "__main__":
    documents = load_documents()
    chunks = create_chunks(documents)
    store_to_database(chunks)
    print("Ingestion complete")