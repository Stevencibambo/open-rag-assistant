# ./chroma.py

import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from typing import List, Dict, Optional, Tuple
import uuid
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

class ChromaWebsiteRAG:
    def __init__(
        self,
        model_name: str = "llama3.2",
        chroma_persist_dir: str = "./chroma_db",
        collection_name: str = "website_documents"
    ):
        """
        Initialize RAG system with ChromaDB persistence
        
        Args:
            model_name: Ollama model for embeddings and generation
            chroma_persist_dir: Directory to persist ChromaDB
            collection_name: Chroma collection name
        """
        self.model_name = model_name
        self.chroma_persist_dir = chroma_persist_dir
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        Path(chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=model_name)
        
        # Initialize or load existing ChromaDB
        self.vector_store = self._init_chroma()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len,
        )
        
        # Track loaded URLs
        self.loaded_urls = set()
        
    def _init_chroma(self) -> Chroma:
        """Initialize or load existing ChromaDB"""
        try:
            # Check if ChromaDB exists
            if os.path.exists(os.path.join(self.chroma_persist_dir, "chroma.sqlite3")):
                print(f"[INFO] Loading existing ChromaDB from {self.chroma_persist_dir}")
                vector_store = Chroma(
                    persist_directory = self.chroma_persist_dir,
                    embedding_function = self.embeddings,
                    collection_name = self.collection_name
                )
            else:
                print("[INFO] Creating new ChromaDB")
                vector_store = Chroma(
                    persist_directory=self.chroma_persist_dir,
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name
                )
            return vector_store
        except Exception as e:
            print(f"[WARNING] Error loading ChromaDB: {e}. Creating new one...")
            return Chroma(
                persist_directory=self.chroma_persist_dir,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
    
    def _generate_document_id(self, url: str, content: str) -> str:
        """ Generate unique ID for a document """
        content_hash = hashlib.md5(content.encode()).hexdigest()
        return f"{url}_{content_hash}"
    
    def _extract_metadata(self, url: str, content: str, chunk_index: int = 0) -> Dict:
        """ Extract metadata for document chunk """
        return {
            "source_url": url,
            "chunk_index": chunk_index,
            "timestamp": datetime.now().isoformat(),
            "content_length": len(content),
            "document_id": self._generate_document_id(url, content),
            "source_type": "website"
        }
    
    def load_website_to_chroma(self, url: str, force_reload: bool = False) -> Tuple[int, List[str]]:
        """
        Load a single website URL into ChromaDB
        
        Args:
            url: Website URL to load
            force_reload: Force reload even if URL already loaded
            
        Returns:
            Tuple of (number_of_chunks, list_of_chunk_ids)
        """
        # Check if URL already loaded
        if not force_reload and url in self.loaded_urls:
            print(f"[WARNING] URL already loaded: {url}")
            return 0, []
        
        print(f"[INFO] Loading website: {url}")
        
        try:
            # Load website content
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            if not documents:
                print(f"[ERROR] No content found at {url}")
                return 0, []
            
            # Process and chunk documents
            all_chunks = []
            chunk_ids = []
            
            for doc in documents:
                # Split document into chunks
                chunks = self.text_splitter.split_text(doc.page_content)
                
                for i, chunk in enumerate(chunks):
                    # Generate unique ID for this chunk
                    chunk_id = f"{url}_chunk_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
                    chunk_ids.append(chunk_id)
                    
                    # Create metadata
                    metadata = self._extract_metadata(url, chunk, i)
                    metadata["chunk_id"] = chunk_id
                    metadata["title"] = doc.metadata.get("title", "")
                    
                    all_chunks.append({
                        "id": chunk_id,
                        "text": chunk,
                        "metadata": metadata
                    })
            
            # Add to ChromaDB
            if all_chunks:
                texts = [chunk["text"] for chunk in all_chunks]
                metadatas = [chunk["metadata"] for chunk in all_chunks]
                ids = [chunk["id"] for chunk in all_chunks]
                
                self.vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                # Persist to disk
                self.vector_store.persist()
                
                # Mark URL as loaded
                self.loaded_urls.add(url)
                
                print(f"[INFO] Loaded {len(all_chunks)} chunks from {url}")
                return len(all_chunks), chunk_ids
            else:
                print(f"[INFO] No chunks created from {url}")
                return 0, []
                
        except Exception as e:
            print(f"[ERROR] Error loading {url}: {str(e)}")
            return 0, []
    
    def load_multiple_websites(self, urls: List[str], force_reload: bool = False) -> Dict:
        """
        Load multiple websites into ChromaDB
        
        Args:
            urls: List of website URLs
            force_reload: Force reload even if URLs already loaded
            
        Returns:
            Dictionary with loading statistics
        """
        results = {
            "total_urls": len(urls),
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "details": {}
        }
        
        for url in urls:
            chunks_loaded, chunk_ids = self.load_website_to_chroma(url, force_reload)
            
            results["details"][url] = {
                "chunks_loaded": chunks_loaded,
                "chunk_ids": chunk_ids,
                "success": chunks_loaded > 0
            }
            
            if chunks_loaded > 0:
                results["successful"] += 1
                results["total_chunks"] += chunks_loaded
            else:
                results["failed"] += 1
        
        print(f"\n📊 Loading Summary:")
        print(f"   Successful: {results['successful']}/{results['total_urls']}")
        print(f"   Failed: {results['failed']}/{results['total_urls']}")
        print(f"   Total chunks: {results['total_chunks']}")
        
        return results
    
    def query_chroma(
        self,
        query: str,
        k: int = 5,
        filter_url: Optional[str] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Query ChromaDB for relevant content
        
        Args:
            query: Search query
            k: Number of results to return
            filter_url: Filter results by specific URL
            
        Returns:
            Tuple of (combined_context, list_of_sources)
        """
        # Prepare filter
        where_filter = None
        if filter_url:
            where_filter = {"source_url": filter_url}
        
        try:
            # Search ChromaDB
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=where_filter
            )
            
            # Format context and sources
            context_parts = []
            sources = []
            
            for i, (doc, score) in enumerate(results):
                source_info = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score,
                    "source_url": doc.metadata.get("source_url", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "Unknown")
                }
                sources.append(source_info)
                
                # Add to context with source citation
                context_parts.append(
                    f"[Source {i+1} from {doc.metadata.get('source_url', 'Unknown')}]\n"
                    f"{doc.page_content}\n"
                )
            
            combined_context = "\n---\n".join(context_parts)
            
            return combined_context, sources
            
        except Exception as e:
            print(f"[ERROR] Error querying ChromaDB: {str(e)}")
            return "", []
    
    def generate_response(
        self,
        query: str,
        context: str,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Generate response using Ollama with RAG context"""
        
        prompt = f"""You are a helpful AI assistant that answers questions based on provided context from websites.

CONTEXT FROM WEBSITES:
{context}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer STRICTLY based on the provided context
2. If the context doesn't contain relevant information, say: "I don't have enough information about that from the provided sources."
3. Cite your sources using format [Source X] when referring to specific information
4. Provide a concise but complete answer
5. If asking for follow-up questions could help, suggest them

ANSWER:"""
        
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate information based on given sources."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the ChromaDB collection"""
        try:
            # Get collection
            collection = self.vector_store._collection
            
            # Count documents
            count = collection.count()
            
            # Get unique sources
            results = collection.get(
                include=["metadatas"],
                limit=count
            )
            
            unique_urls = set()
            if results and results['metadatas']:
                for metadata in results['metadatas']:
                    if metadata and 'source_url' in metadata:
                        unique_urls.add(metadata['source_url'])
            
            return {
                "total_documents": count,
                "unique_sources": len(unique_urls),
                "sources": list(unique_urls),
                "persist_directory": self.chroma_persist_dir,
                "collection_name": self.collection_name
            }
        except Exception as e:
            return {"error": str(e)}
    
    def delete_source(self, url: str) -> int:
        """
        Delete all chunks from a specific URL
        
        Args:
            url: URL to delete
            
        Returns:
            Number of documents deleted
        """
        try:
            collection = self.vector_store._collection
            
            # Find documents from this URL
            results = collection.get(
                where={"source_url": url},
                include=["metadatas"]
            )
            
            if results and results['ids']:
                # Delete documents
                collection.delete(ids=results['ids'])
                
                # Update loaded URLs
                if url in self.loaded_urls:
                    self.loaded_urls.remove(url)
                
                print(f"🗑️ Deleted {len(results['ids'])} documents from {url}")
                return len(results['ids'])
            else:
                print(f"⚠️ No documents found for {url}")
                return 0
                
        except Exception as e:
            print(f"❌ Error deleting source: {str(e)}")
            return 0
    
    def clear_collection(self) -> bool:
        """Clear the entire collection"""
        try:
            collection = self.vector_store._collection
            collection.delete(where={})
            self.loaded_urls.clear()
            print("[ERROR] Cleared entire collection")
            return True
        except Exception as e:
            print(f"[ERROR] Error clearing collection: {str(e)}")
            return False


# Advanced version with conversation memory
class ConversationalChromaRAG(ChromaWebsiteRAG):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conversation_history = []
        self.max_history_length = 10
        
    def chat_with_context(
        self,
        query: str,
        k: int = 4,
        temperature: float = 0.7,
        include_history: bool = True
    ) -> Dict:
        """
        Chat with RAG context and maintain conversation history
        
        Returns:
            Dictionary with response and metadata
        """
        # Get relevant context from ChromaDB
        context, sources = self.query_chroma(query, k=k)
        
        if not context:
            return {
                "response": "I don't have enough information to answer that question. Please load some websites first.",
                "sources": [],
                "context_used": ""
            }
        
        # Prepare conversation history
        history_context = ""
        if include_history and self.conversation_history:
            history_context = "\n\nPrevious conversation:\n"
            for exchange in self.conversation_history[-self.max_history_length:]:
                history_context += f"User: {exchange['user']}\n"
                history_context += f"Assistant: {exchange['assistant']}\n"
        
        # Generate response
        full_context = f"{history_context}\n\nCurrent Context:\n{context}"
        response = self.generate_response(query, full_context, temperature)
        
        # Update conversation history
        self.conversation_history.append({
            "user": query,
            "assistant": response,
            "sources": [s["source_url"] for s in sources],
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        return {
            "response": response,
            "sources": sources,
            "context_used": context,
            "query": query,
            "timestamp": datetime.now().isoformat()
        }
    
    def export_conversation(self, filepath: str = "conversation.json"):
        """Export conversation history to JSON"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.conversation_history, f, indent=2)
            print(f"[INFO] Conversation exported to {filepath}")
            return True
        except Exception as e:
            print(f"[ERROR] Error exporting conversation: {str(e)}")
            return False