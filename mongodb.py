import streamlit as st
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid
import logging
from typing import List, Dict, Any
import PyPDF2
import re
from datetime import datetime
import json

# Load environment variables
load_dotenv()

class PureSemanticSearchMongo:
    def __init__(self):
        self.setup_logging()
        self.init_services()
        self.setup_mongodb_indexes()
    
    def setup_logging(self):
        """Setup basic logging."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def init_services(self):
        """Initialize required services (no LLM needed for pure semantic search)."""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info("Embedding model loaded successfully")
            
            # Initialize MongoDB connection with your specific URL
            mongodb_uri = os.getenv('MONGODB_URI', 'mongodb+srv://akogokennedy:7JpYWCsGK7whPy5m@cluster0.mvm1jsz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
            
            if not mongodb_uri:
                raise ValueError("MONGODB_URI environment variable not set")
            
            self.client = MongoClient(mongodb_uri)
            # Use a specific database name for your semantic search
            self.db = self.client[os.getenv('MONGODB_DATABASE', 'semantic_search_db')]
            self.documents_collection = self.db['documents']
            self.chunks_collection = self.db['chunks']
            
            # Test connection
            self.client.admin.command('ping')
            self.logger.info(f"MongoDB connection established to database: {self.db.name}")
            
        except Exception as e:
            self.logger.error(f"Error initializing services: {e}")
            st.error(f"Failed to initialize services: {e}")
            st.stop()
    
    def setup_mongodb_indexes(self):
        """Create necessary indexes for MongoDB."""
        try:
            # Create text index for keyword search
            self.chunks_collection.create_index([("text", "text")])
            
            # Create compound indexes for efficient queries
            self.chunks_collection.create_index([("doc_id", 1), ("position", 1)])
            self.documents_collection.create_index([("id", 1)], unique=True)
            self.chunks_collection.create_index([("id", 1)], unique=True)
            
            self.logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            self.logger.warning(f"Index creation warning (may already exist): {e}")
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            text = self.clean_extracted_text(text)
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error extracting text from PDF: {e}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        cleaned_text = ' '.join(cleaned_lines)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        return cleaned_text.strip()
    
    def process_and_store_document(self, text: str, title: str = None, file_type: str = "text") -> str:
        """Process document and store in MongoDB with embeddings."""
        doc_id = str(uuid.uuid4())
        
        # Create semantic chunks
        chunks = self.create_semantic_chunks(text)
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts)
        
        # Store document in MongoDB
        document = {
            'id': doc_id,
            'title': title or "Untitled",
            'text': text,
            'file_type': file_type,
            'chunk_count': len(chunks),
            'word_count': len(text.split()),
            'created_at': datetime.utcnow()
        }
        
        self.documents_collection.insert_one(document)
        
        # Store chunks with embeddings
        chunk_documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            chunk_doc = {
                'id': chunk_id,
                'text': chunk['text'],
                'position': i,
                'word_count': chunk['word_count'],
                'doc_id': doc_id,
                'embedding': embedding.tolist(),
                'created_at': datetime.utcnow()
            }
            chunk_documents.append(chunk_doc)
        
        # Bulk insert chunks
        if chunk_documents:
            self.chunks_collection.insert_many(chunk_documents)
        
        return doc_id
    
    def create_semantic_chunks(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[Dict]:
        """Create overlapping chunks that preserve semantic boundaries."""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_word_count = 0
        sentence_buffer = []
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_word_count + sentence_word_count > chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'word_count': current_word_count,
                    'sentences': sentence_buffer.copy()
                })
                
                # Create overlap by keeping last few sentences
                if len(sentence_buffer) > 2:
                    overlap_sentences = sentence_buffer[-2:]
                    current_chunk = ". ".join(overlap_sentences) + ". " + sentence
                    current_word_count = sum(len(s.split()) for s in overlap_sentences) + sentence_word_count
                    sentence_buffer = overlap_sentences + [sentence]
                else:
                    current_chunk = sentence
                    current_word_count = sentence_word_count
                    sentence_buffer = [sentence]
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
                current_word_count += sentence_word_count
                sentence_buffer.append(sentence)
        
        # Add the last chunk
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'word_count': current_word_count,
                'sentences': sentence_buffer
            })
        
        return chunks
    
    def semantic_search(self, query: str, max_results: int = 10, similarity_threshold: float = 0.5) -> Dict[str, Any]:
        """Perform pure semantic search using vector similarity."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Get all chunks and compute similarities (for basic implementation)
        # Note: For production, you'd want to use MongoDB Atlas Vector Search
        chunks_cursor = self.chunks_collection.find({}, {
            'id': 1, 'text': 1, 'position': 1, 'word_count': 1,
            'doc_id': 1, 'embedding': 1
        })
        
        search_results = []
        
        for chunk in chunks_cursor:
            # Calculate cosine similarity
            chunk_embedding = np.array(chunk['embedding'])
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            
            if similarity >= similarity_threshold:
                # Get document info
                document = self.documents_collection.find_one({'id': chunk['doc_id']})
                
                result = {
                    'chunk_id': chunk['id'],
                    'text': chunk['text'],
                    'document_id': chunk['doc_id'],
                    'document_title': document.get('title', 'Untitled') if document else 'Unknown',
                    'file_type': document.get('file_type', 'unknown') if document else 'unknown',
                    'similarity_score': float(similarity),
                    'position': chunk.get('position', 0),
                    'chunk_word_count': chunk.get('word_count', 0),
                    'doc_word_count': document.get('word_count', 0) if document else 0,
                    'relevance_category': self.categorize_relevance(similarity)
                }
                search_results.append(result)
        
        # Sort by similarity score and limit results
        search_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        search_results = search_results[:max_results]
        
        return {
            'query': query,
            'results': search_results,
            'total_results': len(search_results),
            'query_embedding_norm': float(np.linalg.norm(query_embedding))
        }
    
    def keyword_search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform keyword-based fulltext search using MongoDB text search."""
        # Use MongoDB text search
        chunks_cursor = self.chunks_collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(max_results)
        
        search_results = []
        
        for chunk in chunks_cursor:
            # Get document info
            document = self.documents_collection.find_one({'id': chunk['doc_id']})
            
            result = {
                'chunk_id': chunk['id'],
                'text': chunk['text'],
                'document_id': chunk['doc_id'],
                'document_title': document.get('title', 'Untitled') if document else 'Unknown',
                'file_type': document.get('file_type', 'unknown') if document else 'unknown',
                'keyword_score': chunk.get('score', 0),
                'position': chunk.get('position', 0),
                'chunk_word_count': chunk.get('word_count', 0)
            }
            search_results.append(result)
        
        return {
            'query': query,
            'results': search_results,
            'total_results': len(search_results)
        }
    
    def hybrid_search(self, query: str, max_results: int = 10, vector_weight: float = 0.7) -> Dict[str, Any]:
        """Combine semantic (vector) and keyword (fulltext) search."""
        # Semantic search
        semantic_results = self.semantic_search(query, max_results * 2, similarity_threshold=0.3)
        
        # Keyword search
        keyword_results = self.keyword_search(query, max_results * 2)
        
        # Combine and re-rank results
        combined_results = self.combine_search_results(
            semantic_results['results'], 
            keyword_results['results'], 
            vector_weight
        )
        
        return {
            'query': query,
            'results': combined_results[:max_results],
            'total_results': len(combined_results),
            'search_type': 'hybrid',
            'semantic_count': len(semantic_results['results']),
            'keyword_count': len(keyword_results['results'])
        }
    
    def combine_search_results(self, semantic_results: List[Dict], keyword_results: List[Dict], vector_weight: float) -> List[Dict]:
        """Combine and re-rank semantic and keyword search results."""
        # Create a combined results dictionary
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            chunk_id = result['chunk_id']
            result['final_score'] = result['similarity_score'] * vector_weight
            result['has_semantic'] = True
            result['has_keyword'] = False
            combined[chunk_id] = result
        
        # Add/update with keyword results
        keyword_weight = 1.0 - vector_weight
        for result in keyword_results:
            chunk_id = result['chunk_id']
            if chunk_id in combined:
                # Boost score for chunks that appear in both
                combined[chunk_id]['final_score'] += result['keyword_score'] * keyword_weight
                combined[chunk_id]['has_keyword'] = True
                combined[chunk_id]['keyword_score'] = result['keyword_score']
            else:
                # Add keyword-only results
                result['final_score'] = result['keyword_score'] * keyword_weight
                result['has_semantic'] = False
                result['has_keyword'] = True
                result['similarity_score'] = 0.0
                combined[chunk_id] = result
        
        # Sort by final score
        sorted_results = sorted(combined.values(), key=lambda x: x['final_score'], reverse=True)
        return sorted_results
    
    def categorize_relevance(self, score: float) -> str:
        """Categorize relevance based on similarity score."""
        if score >= 0.8:
            return "Highly Relevant"
        elif score >= 0.6:
            return "Relevant"
        elif score >= 0.4:
            return "Moderately Relevant"
        else:
            return "Low Relevance"
    
    def get_document_stats(self) -> Dict[str, int]:
        """Get comprehensive statistics about stored documents."""
        # Get document counts by type
        total_documents = self.documents_collection.count_documents({})
        pdf_documents = self.documents_collection.count_documents({'file_type': 'pdf'})
        text_documents = self.documents_collection.count_documents({'file_type': 'text'})
        legacy_documents = self.documents_collection.count_documents({'file_type': {'$exists': False}})
        
        # Get chunk count
        total_chunks = self.chunks_collection.count_documents({})
        
        # Get word statistics
        pipeline = [
            {'$group': {
                '_id': None,
                'total_words': {'$sum': '$word_count'},
                'avg_words': {'$avg': '$word_count'}
            }}
        ]
        
        word_stats = list(self.documents_collection.aggregate(pipeline))
        total_words = word_stats[0]['total_words'] if word_stats else 0
        avg_words = word_stats[0]['avg_words'] if word_stats else 0
        
        return {
            'total_documents': total_documents,
            'total_chunks': total_chunks,
            'pdf_documents': pdf_documents,
            'text_documents': text_documents,
            'legacy_documents': legacy_documents,
            'total_words': int(total_words),
            'avg_doc_words': int(avg_words)
        }

def main():
    """Main Streamlit application for pure semantic search with MongoDB."""
    st.set_page_config(
        page_title="Pure Semantic Search Engine (MongoDB)",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍  Semantic Search Engine with MongoDB")
    st.markdown("**")
    
    # Initialize the search system
    if 'search_system' not in st.session_state:
        with st.spinner("Initializing semantic search system..."):
            try:
                st.session_state.search_system = PureSemanticSearchMongo()
            except Exception as e:
                st.error(f"Failed to initialize: {e}")
                st.info("Please check your MongoDB connection settings")
                st.stop()
    
    search_system = st.session_state.search_system
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Document Management")
        
        # Display current stats
        try:
            stats = search_system.get_document_stats()
            
            # Stats display
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📄 Total Docs", stats['total_documents'])
                st.metric("🧩 Chunks", stats['total_chunks'])
            with col2:
                st.metric("📝 Words", f"{stats['total_words']:,}")
                if stats['total_documents'] > 0:
                    st.metric("📊 Avg/Doc", f"{stats['avg_doc_words']:,}")
            
            # File type breakdown
            if stats['total_documents'] > 0:
                st.markdown("**File Types:**")
                if stats['pdf_documents'] > 0:
                    st.write(f"📄 PDF: {stats['pdf_documents']}")
                if stats['text_documents'] > 0:
                    st.write(f"📝 Text: {stats['text_documents']}")
                if stats['legacy_documents'] > 0:
                    st.write(f"📋 Legacy: {stats['legacy_documents']}")
                    
        except Exception as e:
            st.error(f"Error loading stats: {e}")
            stats = {'total_documents': 0}
        
        st.markdown("---")
        
        # Document upload
        st.subheader("Add Documents")
        
        # Text input option
        with st.expander("✍️ Add Text Document"):
            doc_title = st.text_input("Document Title")
            doc_text = st.text_area("Document Text", height=200, placeholder="Paste your text content here...")
            
            if st.button("Add Document", type="primary") and doc_text:
                try:
                    with st.spinner("Processing document..."):
                        doc_id = search_system.process_and_store_document(doc_text, doc_title, "text")
                        st.success(f"✅ Document added! ID: {doc_id[:8]}...")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error adding document: {e}")
        
        # File upload option
        with st.expander("📁 Upload Files"):
            uploaded_files = st.file_uploader(
                "Choose files", 
                type=['txt', 'pdf'],
                help="Supported: TXT, PDF - You can select multiple files",
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.info(f"📁 **Selected {len(uploaded_files)} file(s)**")
                
                # Show file details
                total_size = 0
                for file in uploaded_files:
                    total_size += file.size
                    st.write(f"• {file.name} ({file.size / 1024:.1f} KB)")
                
                st.write(f"**Total size:** {total_size / 1024:.1f} KB")
                
                if st.button("Process All Files", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    processed_count = 0
                    failed_count = 0
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        try:
                            status_text.text(f"Processing {uploaded_file.name}...")
                            progress_bar.progress((i) / len(uploaded_files))
                            
                            if uploaded_file.type == "application/pdf":
                                content = search_system.extract_text_from_pdf(uploaded_file)
                                file_type = "pdf"
                            else:
                                content = str(uploaded_file.read(), "utf-8")
                                file_type = "text"
                            
                            doc_id = search_system.process_and_store_document(
                                content, uploaded_file.name, file_type
                            )
                            processed_count += 1
                            
                        except Exception as e:
                            st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
                            failed_count += 1
                    
                    # Final progress update
                    progress_bar.progress(1.0)
                    status_text.text("Processing complete!")
                    
                    # Show results
                    if processed_count > 0:
                        st.success(f"✅ Successfully processed {processed_count} file(s)")
                    if failed_count > 0:
                        st.warning(f"⚠️ Failed to process {failed_count} file(s)")
                    
                    st.rerun()
    
    # Main search interface
    st.header("🔍 Semantic Search")
    
    # Search configuration
    col1, col2, col3 = st.columns([6, 2, 2])
    with col1:
        query = st.text_input(
            "Search Query:",
            placeholder="Enter your search terms or concepts...",
            help="Search uses semantic similarity - related concepts will be found even if exact words don't match"
        )
    
    with col2:
        search_type = st.selectbox("Search Type", ["Semantic", "Keyword", "Hybrid"])
    
    with col3:
        max_results = st.selectbox("Max Results", [5, 10, 20], index=1)
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.3, 0.1)
        with col2:
            if search_type == "Hybrid":
                vector_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.7, 0.1)
    
    # Search execution
    if query:
        try:
            with st.spinner(f"Performing {search_type.lower()} search..."):
                # Perform search based on type
                if search_type == "Semantic":
                    search_results = search_system.semantic_search(query, max_results, similarity_threshold)
                elif search_type == "Keyword":
                    search_results = search_system.keyword_search(query, max_results)
                else:  # Hybrid
                    search_results = search_system.hybrid_search(query, max_results, vector_weight)
                
                if search_results['results']:
                    # Search summary
                    st.markdown(f"### 📊 Search Results ({search_results['total_results']} found)")
                    
                    if search_type == "Hybrid":
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"🔤 Semantic matches: {search_results['semantic_count']}")
                        with col2:
                            st.info(f"🔍 Keyword matches: {search_results['keyword_count']}")
                    
                    # Display search results
                    for i, result in enumerate(search_results['results']):
                        file_type = result.get('file_type', 'unknown')
                        if file_type == 'pdf':
                            file_type_emoji = "📄"
                        elif file_type == 'text':
                            file_type_emoji = "📝"
                        else:
                            file_type_emoji = "📋"  # Legacy or unknown files
                        
                        # Score display
                        if search_type == "Semantic":
                            score_text = f"Similarity: {result['similarity_score']:.3f} ({result['relevance_category']})"
                        elif search_type == "Keyword":
                            score_text = f"Keyword Score: {result['keyword_score']:.3f}"
                        else:  # Hybrid
                            score_text = f"Final: {result['final_score']:.3f}"
                            if result.get('has_semantic') and result.get('has_keyword'):
                                score_text += " (Semantic + Keyword)"
                            elif result.get('has_semantic'):
                                score_text += " (Semantic only)"
                            else:
                                score_text += " (Keyword only)"
                        
                        with st.expander(
                            f"Result {i+1} {file_type_emoji}: {result['text'][:80]}... ({score_text})",
                            expanded=i < 3
                        ):
                            # Metadata
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**Document:** {result['document_title']}")
                                file_type_display = result.get('file_type', 'unknown')
                                if file_type_display and file_type_display != 'unknown':
                                    st.markdown(f"**File Type:** {file_type_display.upper()}")
                                else:
                                    st.markdown(f"**File Type:** LEGACY")
                            with col2:
                                st.markdown(f"**Position:** Chunk {result['position'] + 1}")
                                chunk_words = result.get('chunk_word_count', 'N/A')
                                st.markdown(f"**Words:** {chunk_words}")
                            with col3:
                                if search_type == "Semantic":
                                    st.markdown(f"**Similarity:** {result['similarity_score']:.3f}")
                                    st.markdown(f"**Category:** {result['relevance_category']}")
                                elif search_type == "Hybrid":
                                    if result.get('has_semantic'):
                                        st.markdown(f"**Semantic:** {result['similarity_score']:.3f}")
                                    if result.get('has_keyword'):
                                        st.markdown(f"**Keyword:** {result.get('keyword_score', 'N/A')}")
                            
                            # Content
                            st.markdown("**Content:**")
                            
                            # Highlight search terms in content (for keyword search)
                            content = result['text']
                            if search_type in ["Keyword", "Hybrid"] and result.get('has_keyword'):
                                # Simple highlighting (could be improved)
                                for term in query.split():
                                    content = re.sub(f"({re.escape(term)})", r"**\1**", content, flags=re.IGNORECASE)
                            
                            st.markdown(f"*{content}*")
                else:
                    st.warning("🤷‍♂️ No results found. Try:")
                    st.markdown("- Different search terms")
                    st.markdown("- Lower similarity threshold")
                    st.markdown("- Different search type")
                    st.markdown("- Adding more documents")
        
        except Exception as e:
            st.error(f"Search error: {e}")
    
    # Instructions for new users
    if stats.get('total_documents', 0) == 0:
        st.info("👆 **Get Started:** Add documents using the sidebar to begin searching!")
        
        st.markdown("### 🎯 About Semantic Search")
        st.markdown("""
       
        """)
        
        st.markdown("### 🔧 MongoDB Setup")
        st.markdown("""
       
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("### 💡 Search Tips")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Semantic Search:**
        - Use natural language concepts
        - Related terms will be found automatically
        - Works well for exploratory research
        """)
    
    with col2:
        st.markdown("""
        **Keyword Search:**
        - Use specific terms from documents
        - Good for finding exact phrases
        - MongoDB full-text search capabilities
        """)

if __name__ == "__main__":
    main()