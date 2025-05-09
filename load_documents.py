# ./load_documents.py
# this script serve to load a single document
# for transformation and store the embedding vector in Vector DB
# use ./transformer.py if you want to transform all docs within a
# a given directory

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# 1. load markdown document
loader = UnstructuredMarkdownLoader("docs/public_services.md")
documents = loader.load()

# 2. Split the document in small chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)


# 3. Generating embeddings with Hugging Face Embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Create a vector store
vector_store = Chroma.from_documents(
    texts,
    embedding_function,
    persist_directory="./chroma_db"
)
print("✅ New vector store created successful 🎯")
