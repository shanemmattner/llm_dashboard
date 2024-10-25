import os
import logging
import shutil
import hashlib
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from tqdm import tqdm

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import chromadb
from chromadb.utils import embedding_functions

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingStats:
    """Track document processing statistics"""
    chunks_processed: int = 0
    embedding_time: float = 0.0
    total_time: float = 0.0
    file_size: int = 0
    num_batches: int = 0

class CacheManager:
    """Manage embedding caches"""
    def __init__(self, cache_dir: str = "cache/embeddings"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        cache_key = self.get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
                return None
        return None
    
    def cache_embedding(self, text: str, embedding: List[float]):
        cache_key = self.get_cache_key(text)
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

class DocumentProcessor:
    """Handle document loading and processing"""
    @staticmethod
    def process_text(filepath: str, chunk_size: int = 1024 * 1024) -> List[Document]:
        documents = []
        file_size = os.path.getsize(filepath)
        
        with tqdm(total=file_size, desc="Reading text file", unit='B', unit_scale=True) as pbar:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"source": filepath}
                    ))
                    pbar.update(len(chunk.encode('utf-8')))
        
        return documents

    @staticmethod
    def process_pdf(filepath: str) -> List[Document]:
        try:
            import fitz  # Import PyMuPDF locally
            documents = []
            
            with fitz.open(filepath) as pdf:
                total_pages = len(pdf)
                with tqdm(total=total_pages, desc="Processing PDF pages") as pbar:
                    for page_num in range(total_pages):
                        text = pdf[page_num].get_text()
                        if text.strip():
                            documents.append(Document(
                                page_content=text,
                                metadata={
                                    "source": filepath,
                                    "page": page_num + 1
                                }
                            ))
                        pbar.update(1)
            
            return documents
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            raise

class RAGManager:
    def __init__(self, vector_db_path: str):
        logger.info(f"Initializing RAGManager with path: {vector_db_path}")
        
        self.vector_db_path = vector_db_path
        self.embeddings = OpenAIEmbeddings()
        self.cache_manager = CacheManager()
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=vector_db_path)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        )
        
        # Initialize LLM
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0
        )
        
        logger.info("RAGManager initialization complete")

    def _process_batch(
        self,
        batch: List[Document],
        filename: str,
        start_idx: int,
        stats: ProcessingStats
    ) -> Tuple[List[str], List[Dict], List[str]]:
        """Process a batch of documents"""
        texts = [doc.page_content for doc in batch]
        
        # Get embeddings with caching
        embeddings_start = time.time()
        embeddings_list = []
        
        for text in texts:
            cached_embedding = self.cache_manager.get_cached_embedding(text)
            if cached_embedding:
                embeddings_list.append(cached_embedding)
            else:
                embedding = self.embeddings.embed_query(text)
                self.cache_manager.cache_embedding(text, embedding)
                embeddings_list.append(embedding)
        
        stats.embedding_time += time.time() - embeddings_start
        
        # Prepare batch data
        metadatas = [{
            "source": doc.metadata["source"],
            "chunk_id": f"{filename}_{start_idx+idx}",
            **doc.metadata
        } for idx, doc in enumerate(batch)]
        
        ids = [f"{filename}_{start_idx+idx}" for idx in range(len(batch))]
        
        return texts, metadatas, ids

    def process_file(self, filepath: str, filename: str) -> bool:
        """Process a file and add it to the vector store"""
        try:
            stats = ProcessingStats()
            start_time = time.time()
            stats.file_size = os.path.getsize(filepath)
            
            logger.info(f"Starting processing of {filename} ({stats.file_size/1024/1024:.2f} MB)")
            
            # Load document
            _, ext = os.path.splitext(filepath)
            docs = (DocumentProcessor.process_pdf(filepath) if ext.lower() == '.pdf' 
                   else DocumentProcessor.process_text(filepath))
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=100,
                length_function=len
            )
            splits = text_splitter.split_documents(docs)
            logger.info(f"Split document into {len(splits)} chunks")
            
            # Process in batches
            batch_size = 100
            stats.num_batches = (len(splits) - 1) // batch_size + 1
            
            with tqdm(total=len(splits), desc="Processing chunks") as pbar:
                for i in range(0, len(splits), batch_size):
                    batch = splits[i:i + batch_size]
                    
                    # Process batch
                    texts, metadatas, ids = self._process_batch(batch, filename, i, stats)
                    
                    # Add to ChromaDB
                    self.collection.add(
                        documents=texts,
                        metadatas=metadatas,
                        ids=ids
                    )
                    
                    stats.chunks_processed += len(batch)
                    pbar.update(len(batch))
            
            # Calculate final statistics
            stats.total_time = time.time() - start_time
            
            # Log processing statistics
            logger.info(f"""
            Processing completed:
            - File size: {stats.file_size/1024/1024:.2f} MB
            - Chunks processed: {stats.chunks_processed}
            - Total batches: {stats.num_batches}
            - Embedding time: {stats.embedding_time:.2f}s
            - Total time: {stats.total_time:.2f}s
            - Average speed: {stats.file_size/1024/1024/stats.total_time:.2f} MB/s
            """)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            return False

    def query_documents(self, query: str, selected_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Query the vector store"""
        try:
            logger.info(f"Querying documents with: {query}")
            logger.info(f"Selected files: {selected_files}")
            
            # Query ChromaDB
            where_filter = {"source": {"$in": selected_files}} if selected_files else None
            
            query_start = time.time()
            results = self.collection.query(
                query_texts=[query],
                n_results=10,
                where=where_filter
            )
            query_time = time.time() - query_start
            
            # Convert ChromaDB results to Documents
            documents = []
            if results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    documents.append(Document(
                        page_content=doc,
                        metadata=results['metadatas'][0][i]
                    ))
                logger.info(f"Found {len(documents)} relevant chunks")
                
                # Create context window
                max_context_length = 4000  # Adjust based on model limits
                context = ""
                for doc in documents:
                    if len(context) + len(doc.page_content) + 2 < max_context_length:
                        context += doc.page_content + "\n\n"
                    else:
                        break
                
                # Create prompt
                prompt = f"""Use the following context to answer the question. If you cannot find
                the answer in the context, say so. Do not make up information.
                
                Context:
                {context}
                
                Question: {query}
                
                Answer:"""
                
                # Get response from LLM
                llm_start = time.time()
                response = self.llm.invoke(prompt)
                llm_time = time.time() - llm_start
                
                logger.info(f"Query processed in {query_time:.2f}s, LLM response in {llm_time:.2f}s")
                
                return {
                    'result': response.content,
                    'source_documents': documents,
                    'query_time': query_time,
                    'llm_time': llm_time
                }
            else:
                return {
                    'result': "No relevant documents found to answer the query.",
                    'source_documents': [],
                    'query_time': query_time,
                    'llm_time': 0
                }
            
        except Exception as e:
            logger.error(f"Query error: {e}", exc_info=True)
            raise

    def clear_vectorstore(self) -> bool:
        """Clear the vector store"""
        try:
            logger.info("Starting vector store clear")
            
            # Delete collection
            self.client.delete_collection("documents")
            
            # Remove directory
            if os.path.exists(self.vector_db_path):
                shutil.rmtree(self.vector_db_path)
            
            # Clear cache
            if os.path.exists(self.cache_manager.cache_dir):
                shutil.rmtree(self.cache_manager.cache_dir)
                os.makedirs(self.cache_manager.cache_dir)
            
            # Recreate collection
            self.collection = self.client.get_or_create_collection(
                name="documents",
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            )
            
            logger.info("Vector store cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing store: {e}", exc_info=True)
            return False

    def get_vectorstore_contents(self) -> Dict[str, Any]:
        """Get contents of the vector store"""
        try:
            logger.info("Fetching vector store contents")
            all_metadatas = self.collection.get()['metadatas']
            
            documents_by_source = {}
            for metadata in all_metadatas:
                source = metadata.get('source', 'Unknown')
                if source not in documents_by_source:
                    documents_by_source[source] = {'count': 1}
                else:
                    documents_by_source[source]['count'] += 1
            
            logger.info(f"Found {len(documents_by_source)} documents")
            return documents_by_source
        except Exception as e:
            logger.error(f"Error getting contents: {e}", exc_info=True)
            return {}