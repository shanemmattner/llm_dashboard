import os
import logging
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import chromadb
from chromadb.utils import embedding_functions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handle document loading and processing"""
    
    @staticmethod
    def load_text(filepath: str) -> List[Document]:
        """Load a text file with robust encoding handling"""
        abs_path = os.path.abspath(filepath)
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(abs_path, 'r', encoding=encoding) as f:
                    text = f.read()
                    logger.debug(f"Successfully loaded text file with {encoding}: {abs_path}")
                    return [Document(page_content=text, metadata={"source": abs_path})]
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error loading text file {abs_path}: {e}")
                raise
                
        raise UnicodeDecodeError(f"Failed to decode {abs_path} with any supported encoding")

    @staticmethod
    def load_pdf(filepath: str) -> List[Document]:
        """Load a PDF file with enhanced metadata"""
        abs_path = os.path.abspath(filepath)
        try:
            import fitz  # PyMuPDF
            documents = []
            
            with fitz.open(abs_path) as pdf:
                # Get document metadata
                metadata = pdf.metadata
                total_pages = len(pdf)
                
                logger.debug(f"Processing PDF {abs_path} with {total_pages} pages")
                
                for page_num in range(total_pages):
                    page = pdf[page_num]
                    text = page.get_text()
                    
                    if text.strip():  # Skip empty pages
                        # Enhanced metadata
                        page_metadata = {
                            "source": abs_path,
                            "page": page_num + 1,
                            "total_pages": total_pages,
                            "title": metadata.get("title", ""),
                            "author": metadata.get("author", ""),
                            "creation_date": metadata.get("creationDate", ""),
                            "modification_date": metadata.get("modDate", "")
                        }
                        
                        documents.append(Document(
                            page_content=text,
                            metadata=page_metadata
                        ))
                        
            logger.debug(f"Extracted {len(documents)} pages from PDF")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {abs_path}: {e}")
            raise

class RAGManager:
    """Manage RAG operations and vector store"""
    
    def __init__(self, vector_db_path: str):
        """Initialize RAG Manager with vector store path"""
        self.vector_db_path = os.path.abspath(vector_db_path)
        logger.info(f"RAG Manager initialized with path: {self.vector_db_path}")
        self._setup_components()

    def _setup_components(self):
        """Setup LangChain and ChromaDB components with improved error handling"""
        try:
            # Check API keys
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY environment variable not set")
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            
            # Setup vector store directory
            os.makedirs(self.vector_db_path, exist_ok=True)
            
            # Initialize components with error handling
            self._init_embeddings()
            self._init_chromadb()
            self._init_llm()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise
    
    def _init_embeddings(self):
        """Initialize embeddings with error handling"""
        try:
            self.embeddings = OpenAIEmbeddings()
            logger.debug("OpenAI embeddings initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings: {e}")
            raise
    
    def _init_chromadb(self):
        """Initialize ChromaDB with error handling"""
        try:
            self.client = chromadb.PersistentClient(path=self.vector_db_path)
            logger.debug("ChromaDB client initialized")
            
            self.collection = self.client.get_or_create_collection(
                name="documents",
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="text-embedding-3-small"
                )
            )
            logger.debug("ChromaDB collection initialized")
            
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            raise
    
    def _init_llm(self):
        """Initialize LLM with error handling"""
        try:
            self.llm = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0
            )
            logger.debug("Anthropic LLM initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic LLM: {e}")
            raise

    def process_file(self, filepath: str, filename: str) -> bool:
        """Process a file and add it to the vector store with improved handling"""
        try:
            abs_filepath = os.path.abspath(filepath)
            logger.info(f"Processing file: {filename} ({abs_filepath})")
            
            # Generate file hash for deduplication
            file_hash = self._generate_file_hash(abs_filepath)
            
            # Check for existing version
            existing_docs = self.collection.get(
                where={"file_hash": file_hash}
            )
            
            if existing_docs['ids']:
                logger.info(f"File {filename} already processed with same content. Skipping.")
                return True
            
            # Load and process document
            _, ext = os.path.splitext(filepath)
            docs = (DocumentProcessor.load_pdf(abs_filepath) if ext.lower() == '.pdf' 
                   else DocumentProcessor.load_text(abs_filepath))
            
            # Update metadata
            for doc in docs:
                doc.metadata.update({
                    'source': abs_filepath,
                    'file_hash': file_hash,
                    'processing_timestamp': datetime.now().isoformat()
                })
            
            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(docs)
            
            # Remove existing entries
            self._remove_existing_file_entries(abs_filepath)
            
            # Process in batches
            batch_size = 20
            for i in range(0, len(splits), batch_size):
                batch = splits[i:i + batch_size]
                self._process_batch(batch, filename, file_hash, len(splits))
            
            logger.info(f"Successfully processed {filename} into {len(splits)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            return False

    def _process_batch(
        self, 
        batch: List[Document],
        filename: str,
        file_hash: str,
        total_chunks: int
    ):
        """Process a batch of documents"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Generate IDs and metadata
            ids = [f"{timestamp}_{filename}_{i}" for i in range(len(batch))]
            
            metadatas = [{
                "source": doc.metadata['source'],
                "chunk_id": chunk_id,
                "file_hash": file_hash,
                "timestamp": timestamp,
                "total_chunks": total_chunks,
                "chunk_index": i,
                **doc.metadata
            } for i, (chunk_id, doc) in enumerate(zip(ids, batch))]
            
            # Add to collection
            self.collection.add(
                documents=[doc.page_content for doc in batch],
                metadatas=metadatas,
                ids=ids
            )
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise

    def _generate_file_hash(self, filepath: str) -> str:
        """Generate a hash of file contents"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error generating file hash: {e}")
            return datetime.now().isoformat()

    def _remove_existing_file_entries(self, filepath: str):
        """Remove existing entries for a file"""
        try:
            existing = self.collection.get(
                where={"source": filepath}
            )
            if existing['ids']:
                logger.info(f"Removing {len(existing['ids'])} existing chunks")
                self.collection.delete(
                    ids=existing['ids']
                )
        except Exception as e:
            logger.error(f"Error removing existing entries: {e}")

    def query_documents(
        self, 
        query: str, 
        rag_files: Optional[List[str]] = None,
        context_text: Optional[str] = ""
    ) -> Dict[str, Any]:
        """Query documents with improved context handling"""
        try:
            logger.info(f"Querying with: {query}")
            logger.info(f"Selected RAG files: {rag_files}")
            
            # Verify documents exist
            all_docs = self.get_document_list()
            if not all_docs:
                return {
                    'result': "No documents are currently loaded in the system to query from.",
                    'sources': [],
                    'context_used': 0
                }
            
            # Process file selection
            where_filter = None
            if rag_files:
                normalized_files = [os.path.abspath(f) for f in rag_files]
                available_files = set(os.path.abspath(doc['filepath']) for doc in all_docs)
                valid_files = [f for f in normalized_files if f in available_files]
                
                if valid_files:
                    where_filter = {"source": {"$in": valid_files}}                
                else:
                    return {
                        'result': "None of the selected files are available for querying.",
                        'sources': [],
                        'context_used': 0
                    }
            
            # Query collection
            results = self.collection.query(
                query_texts=[query],
                n_results=1000,
                where=where_filter,
                include=['metadatas', 'documents', 'distances']
            )
            
            if not results['documents'] or not results['documents'][0]:
                return {
                    'result': "No relevant information found in the selected documents.",
                    'sources': [],
                    'context_used': 0
                }
            
            # Build context
            contexts = []
            sources_used = set()
            
            for doc, metadata, distance in zip(
                results['documents'][0], 
                results['metadatas'][0],
                results['distances'][0]
            ):
                if not doc.strip():
                    continue
                
                source = metadata.get('source', 'Unknown')
                source_name = os.path.basename(source)
                page = metadata.get('page', '')
                chunk_index = metadata.get('chunk_index', '')
                relevance = 1 - (distance / 2)
                
                sources_used.add(source)
                
                source_info = [
                    f"Source: {source_name}",
                    f"Page: {page}" if page else "",
                    f"Relevance: {relevance:.2%}",
                    f"Chunk: {chunk_index}" if chunk_index else ""
                ]
                
                context_entry = (
                    f"{'=' * 40}\n"
                    f"{' | '.join(filter(None, source_info))}\n"
                    f"{'-' * 40}\n"
                    f"{doc.strip()}\n"
                )
                
                contexts.append(context_entry)
            
            # Add direct context
            if context_text:
                contexts.append(
                    f"{'=' * 40}\n"
                    f"Additional Context:\n"
                    f"{'-' * 40}\n"
                    f"{context_text.strip()}\n"
                )
            
            # Create prompt
            prompt = f"""Based on the following context, please provide a detailed answer to the question. If the information cannot be found in the context, please say so explicitly.

Question: {query}

Context:
{chr(10).join(contexts)}

Please provide your answer, making sure to:
1. Only use information from the provided context
2. Cite specific sources when referencing information
3. Indicate if any part of the question cannot be answered from the given context"""
            
            # Get response
            response = self.llm.invoke(prompt)
            
            return {
                'result': response.content,
                'sources': results['metadatas'][0],
                'context_used': len(contexts),
                'unique_sources': len(sources_used),
                'relevance_scores': [1 - (d/2) for d in results['distances'][0]]
            }
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise

    def get_document_list(self) -> List[Dict[str, Any]]:
            """Get list of documents in the vector store with enhanced metadata"""
            try:
                results = self.collection.get()
                
                # Group by source file with enhanced metadata
                documents = {}
                for metadata in results['metadatas']:
                    source = metadata.get('source', 'Unknown')
                    if source not in documents:
                        documents[source] = {
                            'chunk_count': 1,
                            'timestamp': metadata.get('timestamp', 'Unknown'),
                            'total_pages': metadata.get('total_pages', None),
                            'file_hash': metadata.get('file_hash', ''),
                            'latest_update': metadata.get('processing_timestamp', ''),
                            'document_metadata': {
                                'title': metadata.get('title', ''),
                                'author': metadata.get('author', ''),
                                'creation_date': metadata.get('creation_date', '')
                            }
                        }
                    else:
                        documents[source]['chunk_count'] += 1
                        # Update timestamp if newer
                        if metadata.get('processing_timestamp', '') > documents[source]['latest_update']:
                            documents[source]['latest_update'] = metadata.get('processing_timestamp')
                
                # Convert to list format with enhanced information
                doc_list = [
                    {
                        'filename': os.path.basename(source),
                        'filepath': source,
                        'file_type': os.path.splitext(source)[1].lower(),
                        **info
                    }
                    for source, info in documents.items()
                ]
                
                # Sort by latest update timestamp
                doc_list.sort(key=lambda x: x['latest_update'], reverse=True)
                
                logger.debug(f"Found {len(doc_list)} documents in vector store")
                return doc_list
                
            except Exception as e:
                logger.error(f"Error getting document list: {e}")
                return []

    def clear_store(self) -> bool:
        """Clear the vector store with safe handling"""
        try:
            logger.info("Clearing vector store")
            
            # Verify collection exists before deleting
            try:
                self.client.get_collection("documents")
                self.client.delete_collection("documents")
                logger.debug("Existing collection deleted")
            except Exception:
                logger.debug("No existing collection to delete")
            
            # Recreate collection
            self.collection = self.client.create_collection(
                name="documents",
                embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name="text-embedding-3-small"
                )
            )
            
            logger.info("Vector store cleared and reinitialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")
            return False

    def get_document_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the documents in the store"""
        try:
            results = self.collection.get()
            
            # Initialize stats dictionary
            stats = {
                'total_documents': 0,
                'total_chunks': len(results['metadatas']),
                'avg_chunks_per_doc': 0,
                'document_types': {},
                'document_sizes': {
                    'small': 0,    # < 10 chunks
                    'medium': 0,   # 10-50 chunks
                    'large': 0     # > 50 chunks
                },
                'processing_dates': set(),
                'metadata_stats': {
                    'with_author': 0,
                    'with_title': 0,
                    'with_creation_date': 0
                }
            }
            
            # Process metadata
            documents = {}
            for metadata in results['metadatas']:
                source = metadata.get('source', '')
                
                # Track unique documents
                if source not in documents:
                    documents[source] = {
                        'chunk_count': 1,
                        'file_type': os.path.splitext(source)[1].lower(),
                        'processing_date': metadata.get('processing_timestamp', '').split('T')[0]
                    }
                else:
                    documents[source]['chunk_count'] += 1
                
                # Track metadata completeness
                if metadata.get('author'):
                    stats['metadata_stats']['with_author'] += 1
                if metadata.get('title'):
                    stats['metadata_stats']['with_title'] += 1
                if metadata.get('creation_date'):
                    stats['metadata_stats']['with_creation_date'] += 1
            
            # Calculate document statistics
            stats['total_documents'] = len(documents)
            
            # Process document types and sizes
            for doc_info in documents.values():
                # Document types
                doc_type = doc_info['file_type']
                stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
                
                # Document sizes
                chunk_count = doc_info['chunk_count']
                if chunk_count < 10:
                    stats['document_sizes']['small'] += 1
                elif chunk_count < 50:
                    stats['document_sizes']['medium'] += 1
                else:
                    stats['document_sizes']['large'] += 1
                
                # Processing dates
                if doc_info['processing_date']:
                    stats['processing_dates'].add(doc_info['processing_date'])
            
            # Calculate averages
            if stats['total_documents'] > 0:
                stats['avg_chunks_per_doc'] = stats['total_chunks'] / stats['total_documents']
            
            # Convert dates to list and sort
            stats['processing_dates'] = sorted(list(stats['processing_dates']))
            
            # Convert metadata stats to percentages
            total_chunks = stats['total_chunks']
            if total_chunks > 0:
                for key in stats['metadata_stats']:
                    stats['metadata_stats'][key] = (
                        stats['metadata_stats'][key] / total_chunks * 100
                    )
            
            logger.debug(f"Generated document stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting document stats: {e}")
            return {}