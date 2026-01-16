from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

DB_PATH = str(Path(__file__).parent / "vector_db")
MODEL = "gpt-4.1-mini"
EMBEDDINGS = OpenAIEmbeddings(model = "text-embedding-3-large")
RETRIEVEL_K = 5
 

SYSTEM_PROMPT="""
You are a knowledgeable and friendly professor representing our university.
You are going to explain the questions that the people are asking about subject lectures etc.
Be polite friendly and if you do not know something, say it.
If the given context is relevant, use the given context to answer questions.
Context:
{context}

Answer the question based on the context above: {question}
"""

def get_relevant_chunks(question):
    """
    Returns most relevant context chunks respect to the question.
    """
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=EMBEDDINGS
    )
    results = vector_store.similarity_search_with_score(query=question, k=RETRIEVEL_K)

    full_context="\n\n ---- \n\n".join([doc.page_content for doc, _score in results])

    return full_context

    
    
    

def answer(question):
    """
    Returns an ai response with using our most relevant chunks from all the documents WITH SYSTEM PROMPT.
    """
    llm = ChatOpenAI(model=MODEL, temperature=0)

    relevant_chunks = get_relevant_chunks(question=question)
    
    prompt = SYSTEM_PROMPT.format(context=relevant_chunks, question=question)

    response = llm.invoke(prompt)

    return response.content , relevant_chunks
    
    

answer1 , chunks = answer("what is cash cow in bcg")
print(answer1)
print(chunks)
