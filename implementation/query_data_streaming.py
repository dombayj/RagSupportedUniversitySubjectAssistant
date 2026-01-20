from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

DB_PATH = str(Path(__file__).parent.parent / "vector_db")
MODEL = "gpt-5-mini"
EMBEDDINGS = OpenAIEmbeddings(model = "text-embedding-3-large")
RETRIEVEL_K = 5

vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=EMBEDDINGS
    )
llm = ChatOpenAI(model=MODEL, temperature=0, streaming=True)


SYSTEM_PROMPT="""
You are a knowledgeable and friendly professor representing our university.
You are going to explain the questions that the people are asking about subject lectures etc.
By the way you currently have the Calculus, Computer Techniques and Architecture, Introduction to Java programmin, Japanese culture and Managment
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
    retriever = vector_store.as_retriever()

    return retriever.invoke(question,k=RETRIEVEL_K)

def combined_messages(question , history):
    """
    A combined questions string for a better rag response (inclued last question as well)
    """
    combined = "\n".join([m["content"] for m in history if m["role"] == "user"])
    return combined + "\n" + question

    
    
    

def stream_answer_question(question, history):
    """
    Returns an ai response with using our most relevant chunks from all the documents WITH SYSTEM PROMPT.
    """
    combined = combined_messages(question=question, history=history) 
    docs= get_relevant_chunks(combined)
    context = "\n\n".join([doc.page_content for doc in docs])
    

    currSystemPrompt = SYSTEM_PROMPT.format(context=context, question=question)
    print(question)

    messages = [SystemMessage(currSystemPrompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(question))

    partial = ""
    for chunk in llm.stream(messages):
        partial += chunk.content
        yield partial, docs
    
    

