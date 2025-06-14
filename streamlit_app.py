#!/usr/bin/env python3
"""
Streamlit application for both Hindi and English semantic search
"""

import streamlit as st
import os
import sys
import json
import time
import math
import numpy as np
import traceback
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

# Custom JSONEncoder to handle NaN values
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)
            return float(obj)
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        return super().default(obj)

# Configuration
LANGUAGES = {
    'english': {
        'name': 'English Semantic Search',
        'description': 'Search NIC codes using natural language in English',
        'flag': '🇺🇸',
        'title': 'English Semantic Search',
        'placeholder': 'Describe a business activity or industry...',
        'examples': 'bakery, software development, wheat cultivation, manufacturing of plastic products'
    },
    'hindi': {
        'name': 'Hindi Semantic Search',
        'description': 'हिंदी में प्राकृतिक भाषा का उपयोग करके NIC कोड खोजें',
        'flag': '🇮🇳',
        'title': 'हिंदी सिमैंटिक सर्च',
        'placeholder': 'अपना खोज प्रश्न यहां दर्ज करें...',
        'examples': 'बेकरी, सॉफ्टवेयर विकास, गेहूं की खेती'
    }
}

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Semantic Search Portal",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def load_custom_css():
    """Load custom CSS for styling to match original UI"""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Hero section styling */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-tagline {
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    /* Language selection buttons */
    .language-selection {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .language-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        cursor: pointer;
        border: 2px solid transparent;
        min-width: 250px;
    }
    
    .language-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        border-color: #667eea;
    }
    
    .language-flag {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .language-name {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .language-desc {
        color: #666;
        font-size: 0.9rem;
    }
    
    /* Search interface styling */
    .search-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    /* Result cards styling */
    .result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        border-color: #667eea;
    }
    
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .rank-badge {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .score-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .score-high { background: linear-gradient(135deg, #10b981, #059669); }
    .score-medium { background: linear-gradient(135deg, #f59e0b, #d97706); }
    .score-low { background: linear-gradient(135deg, #ef4444, #dc2626); }
    
    .result-description {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 1rem;
        line-height: 1.4;
    }
    
    .classification-section {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    .classification-item {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .classification-item:last-child {
        border-bottom: none;
    }
    
    .classification-label {
        font-weight: 600;
        color: #4b5563;
        font-size: 0.9rem;
    }
    
    .classification-value {
        color: #1f2937;
        font-size: 0.9rem;
    }
    
    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e5e7eb;
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        color: #1f2937;
    }
    
    .feature-desc {
        color: #6b7280;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Stats cards */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def initialize_search_engines():
    """Initialize search engines for both languages"""
    if 'search_engines' not in st.session_state:
        st.session_state.search_engines = {}
        
        # Add paths to Python path
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir / 'English'))
        sys.path.insert(0, str(current_dir / 'Hindi'))
        
        # Initialize Hindi search engine
        try:
            from Hindi.hindi_semantic_search import HindiSemanticSearch
            hindi_embeddings = str(current_dir / 'Hindi' / 'output_hindi.json')
            hindi_index = str(current_dir / 'Hindi' / 'hindi_faiss.index')

            # Try to initialize with index first, then embeddings
            if os.path.exists(hindi_index):
                st.write("Initializing Hindi engine with index file...")
                hindi_engine = HindiSemanticSearch(index_path=hindi_index, embeddings_file=None)
                if hindi_engine and hasattr(hindi_engine, 'index') and hindi_engine.index is not None:
                    st.session_state.search_engines['hindi'] = hindi_engine
                    st.success("✅ Hindi engine initialized with index")
                else:
                    st.error("❌ Failed to initialize Hindi engine with index")
                    st.session_state.search_engines['hindi'] = None
            elif os.path.exists(hindi_embeddings):
                st.write("Initializing Hindi engine with embeddings file...")
                hindi_engine = HindiSemanticSearch(embeddings_file=hindi_embeddings, index_path=None)
                if hindi_engine and hasattr(hindi_engine, 'index') and hindi_engine.index is not None:
                    st.session_state.search_engines['hindi'] = hindi_engine
                    st.success("✅ Hindi engine initialized with embeddings")
                else:
                    st.error("❌ Failed to initialize Hindi engine with embeddings")
                    st.session_state.search_engines['hindi'] = None
            else:
                st.error("❌ Neither Hindi index nor embeddings file found")
                st.session_state.search_engines['hindi'] = None
                
        except Exception as e:
            st.error(f"❌ Error initializing Hindi engine: {str(e)}")
            st.session_state.search_engines['hindi'] = None
        
        # Initialize English search engine
        try:
            from sentence_transformers import SentenceTransformer
            from English.faiss_index_manager import FAISSIndexManager
            
            english_json = str(current_dir / 'English' / 'output.json')
            
            if os.path.exists(english_json):
                st.session_state.search_engines['english'] = {
                    'model': SentenceTransformer('all-MiniLM-L6-v2'),
                    'faiss_manager': FAISSIndexManager(json_file_path=english_json)
                }
            else:
                st.session_state.search_engines['english'] = None
        except Exception as e:
            st.session_state.search_engines['english'] = None

def perform_hindi_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Perform search using Hindi search engine"""
    if 'hindi' not in st.session_state.search_engines:
        st.error("Hindi search engine is not initialized")
        return []

    search_engine = st.session_state.search_engines['hindi']
    
    if search_engine is None:
        st.error("Hindi search engine is None")
        return []
    

    # --- ACTUAL SEARCH ---
    try:
        results = search_engine.search(query, top_k=top_k)
        st.write(f"Results returned: {len(results)}")
        return results

    except Exception as e:
        st.error(f"Hindi search error: {str(e)}")
        st.error(traceback.format_exc())
        return []


def perform_english_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Perform search using English search engine"""
    if st.session_state.search_engines.get('english') is None:
        return []
    
    try:
        engine = st.session_state.search_engines['english']
        model = engine['model']
        faiss_manager = engine['faiss_manager']
        
        # Load JSON data if not already loaded
        if not hasattr(faiss_manager, 'json_data') or not faiss_manager.json_data:
            faiss_manager.json_data = faiss_manager.load_json_data()
        
        # Load or create index
        if faiss_manager.index is None:
            if os.path.exists(faiss_manager.index_path):
                faiss_manager.load_index()
            else:
                faiss_manager.build_index()
        
        # Generate query embedding
        query_embedding = model.encode([query])
        
        # Search using FAISS - returns list of tuples (doc_id, similarity_score)
        search_results = faiss_manager.search(query_embedding[0], top_k=top_k * 2)
        
        if not search_results:
            return []
        
        # Get document IDs and similarity scores
        doc_ids = [doc_id for doc_id, _ in search_results]
        similarity_dict = {doc_id: sim for doc_id, sim in search_results}
        
        # Fetch documents from JSON data based on doc_ids
        documents = []
        for doc in faiss_manager.json_data:
            if str(doc.get("_id")) in doc_ids:
                documents.append(doc)
        
        # Format results to match expected structure
        results = []
        for doc in documents:
            try:
                # Include the document if it has valid Class (even if Sub-Class is invalid)
                class_val = doc.get("Class")
                is_valid = False
                
                # Check for valid Sub-Class
                subclass_val = doc.get("Sub-Class")
                if subclass_val and str(subclass_val).strip() and str(subclass_val).lower() != "nan":
                    is_valid = True
                # Check for valid Class if Sub-Class is invalid
                elif class_val and str(class_val).strip() and str(class_val).lower() != "nan":
                    is_valid = True
                    
                # Skip document if neither Class nor Sub-Class is valid
                if not is_valid:
                    continue
                
                # Look up similarity score from our results
                similarity = similarity_dict.get(str(doc["_id"]), 0.0)
                
                # Apply minimum similarity threshold
                if similarity < 0.3:
                    continue
                
                # Format result to match the original semantic search format
                result = {
                    'score': similarity,
                    'rank': len(results) + 1,
                    'id': str(doc.get('_id', '')),
                    'title': doc.get('Description', ''),
                    'section': doc.get('Section', ''),
                    'section_description': doc.get('Section_Description', ''),
                    'division': doc.get('Divison', ''),
                    'division_description': doc.get('Division_Description', ''),
                    'group': doc.get('Group', ''),
                    'group_description': doc.get('Group_Description', ''),
                    'class': doc.get('Class', ''),
                    'class_description': doc.get('Class_Description', ''),
                    'subclass': doc.get('Sub-Class', ''),
                    'subclass_description': doc.get('Sub-Class_Description', ''),
                    'description': doc.get('Description', ''),
                    'similarity': similarity,
                    'similarity_percent': int(similarity * 100) if isinstance(similarity, float) else similarity,
                    # Add fields to match original semantic search format
                    'NIC_Code': doc.get('Sub-Class', 'N/A'),
                    'Section': doc.get('Section', 'N/A'),
                    'Division': doc.get('Divison', 'N/A'),
                    'Group': doc.get('Group', 'N/A'),
                    'Class': doc.get('Class', 'N/A'),
                    'Sub-Class': doc.get('Sub-Class', 'N/A'),
                    'Description': doc.get('Description', '')
                }
                
                results.append(result)
                
            except Exception as e:
                st.warning(f"Error processing document {doc.get('_id', 'unknown')}: {e}")
                continue
        
        # Sort by similarity score (descending) and limit to top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
        
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def format_field_value(value):
    """Format field values for display"""
    if value is None or value == '' or str(value).lower() in ['nan', 'null', 'none']:
        return "Not specified"
    return str(value)

def create_results_visualization(results: List[Dict[str, Any]]):
    """Create a bar chart visualization of search results"""
    if not results:
        return
    
    # Prepare data for visualization
    labels = []
    scores = []
    
    for i, result in enumerate(results[:10]):  # Show top 10 results
        doc = result.get('document', result)
        description = doc.get('description', f'Result {i+1}')
        
        # Truncate long descriptions
        if len(description) > 30:
            description = description[:30] + '...'
        
        labels.append(description)
        
        score = result.get('score', 0)
        score_percentage = int(score * 100) if isinstance(score, float) else score
        scores.append(score_percentage)
    
    # Create plotly bar chart
    fig = px.bar(
        x=scores,
        y=labels,
        orientation='h',
        title="Top Search Results by Similarity Score",
        labels={'x': 'Similarity Score (%)', 'y': 'Description'},
        color=scores,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_font_size=16,
        font=dict(family="Poppins, sans-serif")
    )
    
    return fig

def display_search_results(results: List[Dict[str, Any]], language: str):
    """Display search results with proper formatting"""
    if not results:
        st.warning("No results found. Try a different search term.")
        return
    
    st.success(f"Found {len(results)} results")
    
    # Create visualization
    if len(results) > 1:
        st.subheader("🔍 Search Results Visualization")
        fig = create_results_visualization(results)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    
    # Display results
    st.subheader("📊 Detailed Results")
    
    for i, result in enumerate(results):
        # Get document data based on language
        if language == 'hindi':
            doc = result.get('document', {})
            score = result.get('score', 0)
            rank = result.get('rank', i + 1)
        else:
            doc = result
            score = result.get('score', 0)
            rank = i + 1
        
        # Calculate score percentage and color
        score_percentage = int(score * 100) if isinstance(score, float) else score
        
        # Get description or use NIC code as fallback
        description = doc.get('description', 'No description available')
        nic_code = format_field_value(doc.get('subclass'))
        
        if nic_code and nic_code != 'N/A' and nic_code.strip():
            # NIC code is available - show both
            if description and description != 'No description available' and description.strip():
                display_title = f"🔢 **{nic_code}** - {description}"
            else:
                display_title = f"🔢 **{nic_code}** - *No description available*"
        else:
            # NIC code is not available - show only description
            if description and description != 'No description available' and description.strip():
                display_title = f"{description}"
            else:
                display_title = f"*No description available*"
        
        # Use Streamlit native components instead of HTML
        with st.container():
            col1, col2 = st.columns([1, 10])
            
            with col1:
                # Score badge
                if score_percentage >= 80:
                    st.success(f"#{rank}")
                    st.success(f"{score_percentage}%")
                elif score_percentage >= 50:
                    st.warning(f"#{rank}")
                    st.warning(f"{score_percentage}%")
                else:
                    st.error(f"#{rank}")
                    st.error(f"{score_percentage}%")
            
            with col2:
                # Title/Description
                st.write(f"**{display_title}**")
                
                # If we used NIC code as title, show any available description below
                if display_title.startswith("NIC Code:") and description != 'No description available':
                    st.write(f"*{description}*")
                
                # Classification details
                st.write("**Classification Details:**")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"📑 **Section:** {format_field_value(doc.get('section'))}")
                    st.write(f"📋 **Division:** {format_field_value(doc.get('division'))}")
                    st.write(f"👥 **Group:** {format_field_value(doc.get('group'))}")
                
                with col_b:
                    st.write(f"🏷️ **Class:** {format_field_value(doc.get('class'))}")
                    st.write(f"🏷️ **Sub-Class:** {format_field_value(doc.get('subclass'))}")
                    st.write(f"🔢 **NIC Code:** {nic_code}")
        
        st.divider()

def render_language_interface(language: str):
    """Render the search interface for the selected language"""
    config = LANGUAGES[language]
    
    # Hero section
    hero_html = f"""
    <div class="hero-section">
        <div class="hero-title">{config['flag']} {config['title']}</div>
        <div class="hero-tagline">Discover semantically similar documents with AI-powered search</div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)
    
    # Search interface
    # st.markdown('<div class="search-section">', unsafe_allow_html=True)
    
    # Search form
    with st.form(key=f"search_form_{language}", clear_on_submit=False):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            query = st.text_input(
                "Search Query",
                placeholder=config['placeholder'],
                key=f"query_{language}",
                label_visibility="collapsed"
            )
        
        with col2:
            top_k = st.selectbox(
                "Results",
                [5, 10, 20, 50],
                index=1,
                key=f"top_k_{language}"
            )
        
        search_button = st.form_submit_button("🔍 Search", use_container_width=True)
    
    # st.markdown('</div>', unsafe_allow_html=True)
    
    # Example searches
    st.markdown(f"**Example searches:** {config['examples']}")
    
    # Perform search when button is clicked
    if search_button and query.strip():
        with st.spinner('🔍 Searching...'):
            start_time = time.time()
            
            if language == 'hindi':
                results = perform_hindi_search(query, top_k)
            else:
                results = perform_english_search(query, top_k)
            
            search_time = (time.time() - start_time) * 1000
            
            # Display search stats
            st.info(f"⚡ Search completed in {search_time:.2f} ms")
            
            # Display results
            display_search_results(results, language)

def render_main_page():
    """Render the main page with language selection"""
    # Hero section
    hero_html = """
    <div class="hero-section">
        <div class="hero-title">🔍 Semantic Search Portal</div>
        <div class="hero-tagline">Search for industry codes and descriptions using natural language</div>
        <div style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
            Powered by FAISS and advanced NLP models
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)
    
    # Language selection
    st.markdown("## 🌐 Choose Your Language")
    
    # Language cards
    col1, col2 = st.columns(2)
    
    with col1:
        english_card = f"""
        <div class="language-card" onclick="selectLanguage('english')">
            <div class="language-flag">{LANGUAGES['english']['flag']}</div>
            <div class="language-name">English</div>
            <div class="language-desc">{LANGUAGES['english']['description']}</div>
        </div>
        """
        st.markdown(english_card, unsafe_allow_html=True)
        
        if st.button("Select English", key="english_select", use_container_width=True):
            st.session_state.selected_language = 'english'
            st.rerun()
    
    with col2:
        hindi_card = f"""
        <div class="language-card" onclick="selectLanguage('hindi')">
            <div class="language-flag">{LANGUAGES['hindi']['flag']}</div>
            <div class="language-name">Hindi / हिंदी</div>
            <div class="language-desc">{LANGUAGES['hindi']['description']}</div>
        </div>
        """
        st.markdown(hindi_card, unsafe_allow_html=True)
        
        if st.button("Select Hindi", key="hindi_select", use_container_width=True):
            st.session_state.selected_language = 'hindi'
            st.rerun()
    
    # Features section
    st.markdown("---")
    st.markdown("### ✨ Features")
    
    features_html = """
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <div class="feature-title">Semantic Search</div>
            <div class="feature-desc">Natural language queries with AI-powered understanding and contextual matching</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Fast Performance</div>
            <div class="feature-desc">FAISS vector indexing with optimized search algorithms for real-time results</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🌍</div>
            <div class="feature-title">Multi-language</div>
            <div class="feature-desc">English and Hindi support with native language processing and cultural context</div>
        </div>
    </div>
    """
    st.markdown(features_html, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    setup_page_config()
    load_custom_css()
    
    # Initialize session state
    if 'selected_language' not in st.session_state:
        st.session_state.selected_language = None
    
    # Initialize search engines
    if 'engines_initialized' not in st.session_state:
        initialize_search_engines()
        st.session_state.engines_initialized = True
    
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.selected_language = None
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("## 🌐 Languages")
        
        if st.button("🇮🇳 English", use_container_width=True):
            st.session_state.selected_language = 'english'
            st.rerun()
        
        if st.button("🇮🇳 Hindi", use_container_width=True):
            st.session_state.selected_language = 'hindi'
            st.rerun()
        
        st.markdown("---")
        
        # System status
        st.markdown("## ⚙️ System Status")
        
        if hasattr(st.session_state, 'search_engines'):
            if st.session_state.search_engines.get('english'):
                st.success("✅ English Engine Ready")
            else:
                st.error("❌ English Engine Failed")
            
            if st.session_state.search_engines.get('hindi'):
                st.success("✅ Hindi Engine Ready")
            else:
                st.error("❌ Hindi Engine Failed")
        
        st.markdown("---")
        
        st.markdown("## ℹ️ About")
        st.markdown("""
        This application provides semantic search capabilities for NIC codes in both English and Hindi.
        
        **Technologies:**
        - FAISS for vector search
        - Transformers for embeddings
        - Streamlit for UI
        
        **Features:**
        - Real-time search
        - Multilingual support
        - Interactive visualizations
        """)
    
    # Main content area
    if st.session_state.selected_language is None:
        render_main_page()
    else:
        # Show back button
        if st.button("← Back to Language Selection", key="back_button"):
            st.session_state.selected_language = None
            st.rerun()
        
        render_language_interface(st.session_state.selected_language)

if __name__ == "__main__":
    main()
