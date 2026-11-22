# ./transformer-url.py

import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# URLs to index (WHITELIST)
URLS = [
    {
        "url": "https://www.kadea.academy/formations/dev-web/?city=kinshasa",
        "category": "content",
        "training": "Web Developer"
    },
    {
        "url": "https://www.kadea.academy/formations/dev-ia/?city=kinshasa",
        "category": "content",
        "training": "IA Developer"
    },
    {
        "url": "https://www.kadea.academy/formations/marketing-digital/?city=kinshasa",
        "category": "content",
        "training": "Digital Marketing"
    }
]

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

# Embedding model (same as documents)
embedding_function = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

def load_website_documents(urls):
    documents = []

    for entry in urls:
        url = entry["url"]
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
        except Exception as e:
            print(f"[ERROR] Failed to fetch {url}: {e}")
            continue

        soup = BeautifulSoup(res.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        doc = Document(
            page_content=text,
            metadata={
                "source": "website",
                "url": url,
                "category": entry.get("category"),
                "training": entry.get("training")
            }
        )

        documents.append(doc)

    return documents


print("[INFO] Loading website pages...")
documents = load_website_documents(URLS)
print(f"[INFO] {len(documents)} web documents loaded.")

# Split like documents
texts = splitter.split_documents(documents)

# Store in SAME Chroma DB (or another if you prefer)
vector_store = Chroma.from_documents(
    texts,
    embedding_function,
    persist_directory="./chroma_db"
)

print("[INFO] Website vector store created and persisted!")
