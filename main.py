import streamlit as st
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np
import uuid
import logging
from typing import List, Dict, Any
import PyPDF2
import re

# Load environment variables
load_dotenv()

class PureSemanticSearch:
    def __init__(self):
        self.setup_logging()
        self.init_services()
        self.setup_neo4j_schema()
    
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
            
            # Initialize Neo4j connection
            self.driver = GraphDatabase.driver(
                os.getenv('NEO4J_URI'),
                auth=(os.getenv('NEO4J_USER'), os.getenv('NEO4J_PASSWORD'))
            )
            self.logger.info("Neo4j connection established")
            
        except Exception as e:
            self.logger.error(f"Error initializing services: {e}")
            st.error(f"Failed to initialize services: {e}")
            st.stop()
    
    def setup_neo4j_schema(self):
        """Create necessary indexes and constraints."""
        with self.driver.session() as session:
            try:
                # Create constraints
                session.run("CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
                session.run("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
                
                # Create vector index for embeddings
                session.run("""
                    CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
                    FOR (c:Chunk) ON (c.embedding)
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 384,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)
                
                # Create text index for keyword search
                session.run("CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS FOR (c:Chunk) ON EACH [c.text]")
                
                self.logger.info("Neo4j schema setup completed")
                
            except Exception as e:
                self.logger.warning(f"Schema setup warning (may already exist): {e}")
    
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
        """Process document and store in Neo4j with embeddings."""
        doc_id = str(uuid.uuid4())
        
        # Create semantic chunks
        chunks = self.create_semantic_chunks(text)
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts)
        
        # Store in Neo4j
        with self.driver.session() as session:
            # Create document node
            session.run("""
                CREATE (d:Document {
                    id: $doc_id,
                    title: $title,
                    text: $text,
                    file_type: $file_type,
                    chunk_count: $chunk_count,
                    word_count: $word_count,
                    created_at: datetime()
                })
            """, 
            doc_id=doc_id, 
            title=title or "Untitled", 
            text=text, 
            file_type=file_type, 
            chunk_count=len(chunks),
            word_count=len(text.split())
            )
            
            # Create chunk nodes with embeddings
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{doc_id}_chunk_{i}"
                
                session.run("""
                    CREATE (c:Chunk {
                        id: $chunk_id,
                        text: $text,
                        position: $position,
                        word_count: $word_count,
                        doc_id: $doc_id,
                        embedding: $embedding
                    })
                    WITH c
                    MATCH (d:Document {id: $doc_id})
                    CREATE (d)-[:CONTAINS {position: $position}]->(c)
                """, 
                chunk_id=chunk_id,
                text=chunk['text'],
                position=i,
                word_count=chunk['word_count'],
                doc_id=doc_id,
                embedding=embedding.tolist()
                )
                
                # Create sequential relationships between chunks
                if i > 0:
                    prev_chunk_id = f"{doc_id}_chunk_{i-1}"
                    session.run("""
                        MATCH (prev:Chunk {id: $prev_chunk_id})
                        MATCH (curr:Chunk {id: $curr_chunk_id})
                        CREATE (prev)-[:FOLLOWS]->(curr)
                    """, prev_chunk_id=prev_chunk_id, curr_chunk_id=chunk_id)
        
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
        
        # Search for similar chunks
        with self.driver.session() as session:
            results = session.run("""
                CALL db.index.vector.queryNodes('chunk_embeddings', $max_results, $query_embedding)
                YIELD node, score
                WHERE score >= $similarity_threshold
                MATCH (d:Document)-[:CONTAINS]->(node)
                
                RETURN node.id as chunk_id,
                       node.text as text,
                       node.position as position,
                       node.word_count as chunk_word_count,
                       d.id as document_id,
                       d.title as document_title,
                       COALESCE(d.file_type, 'legacy') as file_type,
                       COALESCE(d.word_count, 0) as doc_word_count,
                       score as similarity_score
                ORDER BY score DESC
            """, 
            query_embedding=query_embedding.tolist(), 
            max_results=max_results,
            similarity_threshold=similarity_threshold
            )
            
            search_results = []
            for record in results:
                result = {
                    'chunk_id': record['chunk_id'],
                    'text': record['text'],
                    'document_id': record['document_id'],
                    'document_title': record['document_title'],
                    'file_type': record['file_type'],
                    'similarity_score': record['similarity_score'],
                    'position': record['position'] or 0,
                    'chunk_word_count': record['chunk_word_count'] or 0,
                    'doc_word_count': record['doc_word_count'],
                    'relevance_category': self.categorize_relevance(record['similarity_score'])
                }
                search_results.append(result)
        
        return {
            'query': query,
            'results': search_results,
            'total_results': len(search_results),
            'query_embedding_norm': float(np.linalg.norm(query_embedding))
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
    
    def keyword_search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Perform keyword-based fulltext search."""
        with self.driver.session() as session:
            results = session.run("""
                CALL db.index.fulltext.queryNodes('chunk_fulltext', $search_query)
                YIELD node, score
                MATCH (d:Document)-[:CONTAINS]->(node)
                RETURN node.id as chunk_id,
                       node.text as text,
                       node.position as position,
                       COALESCE(node.word_count, 0) as chunk_word_count,
                       d.id as document_id,
                       d.title as document_title,
                       COALESCE(d.file_type, 'legacy') as file_type,
                       score as keyword_score
                ORDER BY score DESC
                LIMIT $max_results
            """, search_query=query, max_results=max_results)
            
            search_results = []
            for record in results:
                result = {
                    'chunk_id': record['chunk_id'],
                    'text': record['text'],
                    'document_id': record['document_id'],
                    'document_title': record['document_title'],
                    'file_type': record['file_type'],
                    'keyword_score': record['keyword_score'],
                    'position': record['position'] or 0,
                    'chunk_word_count': record['chunk_word_count']
                }
                search_results.append(result)
        
        return {
            'query': query,
            'results': search_results,
            'total_results': len(search_results)
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
        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
                RETURN count(DISTINCT d) as total_documents,
                       count(c) as total_chunks,
                       count(DISTINCT CASE WHEN d.file_type = 'pdf' THEN d END) as pdf_documents,
                       count(DISTINCT CASE WHEN d.file_type = 'text' THEN d END) as text_documents,
                       count(DISTINCT CASE WHEN d.file_type IS NULL THEN d END) as legacy_documents,
                       sum(d.word_count) as total_words,
                       avg(d.word_count) as avg_doc_words
            """)
            
            record = result.single()
            return {
                'total_documents': record['total_documents'],
                'total_chunks': record['total_chunks'],
                'pdf_documents': record['pdf_documents'] or 0,
                'text_documents': record['text_documents'] or 0,
                'legacy_documents': record['legacy_documents'] or 0,
                'total_words': int(record['total_words'] or 0),
                'avg_doc_words': int(record['avg_doc_words'] or 0)
            }

def main():
    """Main Streamlit application for pure semantic search."""
    st.set_page_config(
        page_title="Semantic Search Engine",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Pure Semantic Search Engine with Neo4j")
    st.markdown("*Vector-based semantic similarity search with optional keyword matching*")
    
    # Initialize the search system
    if 'search_system' not in st.session_state:
        with st.spinner("Initializing semantic search system..."):
            st.session_state.search_system = PureSemanticSearch()
    
    search_system = st.session_state.search_system
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Document Management")
        
        # Display current stats
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
        
        st.markdown("---")
        
        # Document upload
        st.subheader("Add Documents")
        
        # Text input option
        with st.expander("✍️ Add Text Document"):
            doc_title = st.text_input("Document Title")
            doc_text = st.text_area("Document Text", height=200, placeholder="Paste your text content here...")
            
            if st.button("Add Document", type="primary") and doc_text:
                with st.spinner("Processing document..."):
                    doc_id = search_system.process_and_store_document(doc_text, doc_title, "text")
                    st.success(f"✅ Document added! ID: {doc_id[:8]}...")
                    st.rerun()
        
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
    
    # Instructions for new users
    if stats['total_documents'] == 0:
        st.info("👆 **Get Started:** Add documents using the sidebar to begin searching!")
        
        st.markdown("### 🎯 About Semantic Search")
        st.markdown("""
        **Semantic search** finds content based on meaning and context, not just exact word matches:
        
        - **Query:** "machine learning algorithms"
        - **Finds:** Documents about "neural networks", "deep learning", "AI models"
        - **Even if** those exact words aren't in your query
        
        **Search Types:**
        - **Semantic:** Uses vector embeddings for conceptual similarity
        - **Keyword:** Traditional text matching
        - **Hybrid:** Combines both approaches for comprehensive results
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
        - Case-insensitive matching
        """)

if __name__ == "__main__":
    main()