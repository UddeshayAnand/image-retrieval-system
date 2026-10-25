import streamlit as st
import base64
from PIL import Image as PILImage
import os

def load_css(file_name):
    """Load external CSS file."""
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def image_card(image_path, confidence=None, distance=None, rank=None):
    """
    Displays an image in a styled card with search results metadata.
    Matches the Colab notebook's display format.
    """
    with st.container():
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        
        # Load and display image
        try:
            if os.path.exists(image_path):
                img = PILImage.open(image_path)
                st.image(img, use_container_width=True)
            else:
                st.warning(f"Image not found: {image_path}")
        except Exception as e:
            st.error(f"Error loading image: {e}")
        
        # Display metadata
        if rank is not None:
            st.markdown(f"**Rank #{rank}**")
        
        if confidence is not None:
            st.metric("Confidence", f"{confidence:.4f}")
        
        if distance is not None:
            st.metric("Distance", f"{distance:.4f}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def sidebar_info():
    """Display information panel in sidebar."""
    st.sidebar.title("Image Retrieval System")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("System Info")
    
    info_text = """
    **Feature Extractor**: ResNet50  
    **Embedding Dim**: 256D (2048D PCA)  
    **Clustering**: K-Means (K=300)  
    **Distance Metric**: Euclidean  
    **Dataset**: 100K Tiny ImageNet  
    **Search Strategy**: Cluster-based
    """
    
    st.sidebar.info(info_text)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("How It Works")
    
    steps = """
    1. Upload an image
    2. Extract 2048D features via ResNet50
    3. Reduce to 256D using PCA
    4. Find cluster using K-Means (K=300)
    5. Calculate Euclidean distances
    6. Rank by confidence score
    7. Display top-5 results
    """
    
    st.sidebar.markdown(steps)
