# ./transformer.py
# this script serve to transform all docs within the given directory

import os
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    TextLoader,
    CSVLoader,
    UnstructuredFileLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Documents directory folder
DOCS_DIR = "docs/"

# Supported extensions and corresponding loaders
EXTENSION_LOADER_MAPPING = {
    ".pdf": UnstructuredPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".md": UnstructuredMarkdownLoader,
}


# Function to dynamically load all files from the given directory
def load_documents_from_directory(directory):
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        ext = os.path.splitext(file_path)[1].lower()
        try:
            load_class = EXTENSION_LOADER_MAPPING.get(ext, UnstructuredFileLoader)
            loader = load_class(file_path)
            docs = loader.load()
            documents.extend(docs)
        except Exception as e:
            print(f"❌ Failed to load {file_path}: {e}")
    return documents


# Load all documents
print("📁 Loading documents...")
documents = load_documents_from_directory(DOCS_DIR)
print(f"✅ {len(documents)} documents loaded.")

# Split documents in small chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = splitter.split_documents(documents)

# Embedding model
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Vector store
vector_store = Chroma.from_documents(
    texts,
    embedding_function,
    persist_directory="./chroma_db"
)
print("✅ Vector store created and persisted!")


