# ./agent.py
# here the main point of the system

from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# Initialize the LLM e.g: llama3.2 or mistral
llm = Ollama(model="llama3.2")

# Initialize Memory
memory = ConversationBufferMemory(input_key="chat_history")

# Memory already exists (ConversionBufferMemory)
chat_history = []

# Load Vector Store
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_function
)

# Build retrial-based QA
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(),
    memory=memory
)

# Build a conversation chain
# conversation = ConversationChain(llm=llm, memory=memory, verbose=True)

# Define small talk triggers
SMALL_TALK_KEYWORDS = [
    "merci", "bonjour", "thank you", "hello", "hi", "ça va", "salut", "bonjour", "ok", "cool", "thanks"
]


def is_small_talk(question: str) -> bool:
    question_lower = question.lower().strip()
    # if the question a toot short (e.g: "hi", "merci", "bonjour")
    if len(question_lower.split()) <= 4:
        for keyword in SMALL_TALK_KEYWORDS:
            if keyword in question_lower:
                return True
    return False


def ask_agent(question: str):
    global chat_history
    if is_small_talk(question):
        # small message -> forward message directly to the llm without RAG
        response = llm.invoke(question)
    else:
        result = qa_chain.invoke({"question": question, "chat_history": chat_history})
        response = result['answer']

    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(response)
    return response
