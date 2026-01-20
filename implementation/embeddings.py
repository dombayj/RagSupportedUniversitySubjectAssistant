import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
load_dotenv(override=True)

DATA_PATH= str(Path(__file__).parent.parent / "subject_documents")
DB_PATH = str(Path(__file__).parent.parent / "vector_db")
MODEL = "gpt-4.1-mini"
EMBEDDINGS = OpenAIEmbeddings(model = "text-embedding-3-large")


#LOAD PDFS
def load_documents():
    folders = glob.glob(str(Path(DATA_PATH) / "*"))
    documents=[]
    for folder in folders:
        doc_type = os.path.basename(folder)
        # if path
        loaderPdf = DirectoryLoader(
            path=folder,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
            )
        loaderPptx=DirectoryLoader(
            path=folder,
            glob="**/*.pptx",
            loader_cls=UnstructuredPowerPointLoader
            )
        folder_doc = loaderPdf.load() + loaderPptx.load()
        for doc in folder_doc:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents
            


#SPLIT INTO CHUNKS
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 800 ,chunk_overlap = 200,length_function=len,is_separator_regex=False)
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_chunks_with_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index+= 1
        else:
            current_chunk_index = 0

        
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
    return chunks

#VECTORIZE
def store_to_database(chunks):
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=EMBEDDINGS
    )

    chunks_with_ids = create_chunks_with_ids(chunks=chunks)

    existing_items = vector_store.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks=[]
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)


    if len(new_chunks):
        print(f"Adding {len(new_chunks)} new documents")
        new_chunks_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vector_store.add_documents(documents=new_chunks, ids=new_chunks_ids)
    else:
        print("No documents to add to database")

    
if __name__ == "__main__":
    documents = load_documents()
    chunks = create_chunks(documents=documents)
    chunks = create_chunks_with_ids(chunks=chunks)
    store_to_database(chunks=chunks)


