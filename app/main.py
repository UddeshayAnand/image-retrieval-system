import streamlit as st
import os
import sys
from PIL import Image as PILImage

# Add project root to path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ui_components import load_css, sidebar_info
from src.inference import InferenceEngine
from src.utils import load_image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Page Config
st.set_page_config(
    page_title="Visual Search & Clustering",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Styles
load_css("app/styles.css")

# Initialize Engine
@st.cache_resource
def get_engine():
    return InferenceEngine()

# Clear cache button (for development/updates)
if st.sidebar.button("🔄 Refresh Engine Cache"):
    st.cache_resource.clear()
    st.rerun()

engine = get_engine()

# Sidebar
sidebar_info()
mode = st.sidebar.radio("Mode", ["Search", "Cluster Analysis"])

# Main Content
st.title("Visual Search & Clustering System")
st.markdown("---")

if mode == "Search":
    st.subheader("Image Retrieval - Find Similar Images")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Upload Query Image")
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            if image:
                st.image(image, caption="Query Image", use_container_width=True)
                
                if st.button("Search Similar Images", key="search_btn"):
                    with st.spinner("Extracting features and searching..."):
                        results = engine.search(image, k=5)
                    
                    if results:
                        st.success(f"Found {len(results)} similar images!")
                        
                        # Display results in a table format first
                        st.markdown("### Search Results")
                        
                        # Create results dataframe
                        results_data = []
                        for res in results:
                            results_data.append({
                                'Rank': res['rank'],
                                'Confidence': f"{res['confidence']:.4f}",
                                'Distance': f"{res['distance']:.4f}",
                                'Filename': res['filename']
                            })
                        
                        df_results = pd.DataFrame(results_data)
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Display images in grid
                        st.markdown("### Similar Images")
                        cols = st.columns(3)
                        
                        for idx, res in enumerate(results):
                            col = cols[idx % 3]
                            with col:
                                try:
                                    if os.path.exists(res['path']):
                                        img = PILImage.open(res['path'])
                                        st.image(
                                            img,
                                            caption=f"Rank {res['rank']}: {res['filename']}\nConfidence: {res['confidence']:.4f}\nDistance: {res['distance']:.4f}",
                                            use_container_width=True
                                        )
                                    else:
                                        st.warning(f"Image not found: {res['path']}")
                                except Exception as e:
                                    st.error(f"Error loading image: {e}")
                    else:
                        st.warning("No similar images found. Try a different image.")

    with col2:
        if uploaded_file is None:
            st.info("Upload an image on the left to search for similar images.")
            st.markdown("""
            ### How It Works
            
            1. **Upload** any image
            2. **Extract Features**: ResNet50 extracts 2048D features, PCA reduces to 256D
            3. **Find Cluster**: KMeans (K=300) predicts image cluster
            4. **Calculate Distance**: Euclidean distance to all images in cluster
            5. **Rank Results**: Top-5 images sorted by distance
            6. **Confidence Score**: 1.0 - (distance / max_distance_in_cluster)
            
            ### Key Metrics
            - **Confidence**: Higher is more similar (0.0-1.0)
            - **Distance**: Lower is more similar (Euclidean)
            - **Dataset**: 100,000 Tiny ImageNet images
            - **Clusters**: 300 clusters (K=300)
            - **Features**: ResNet50 (2048D) → PCA (256D)
            
            ### Algorithm Details
            - **Feature Extraction**: ResNet50 backbone
            - **Dimensionality**: 2048D → 256D (PCA)
            - **Clustering**: K-Means (K=300, optimal from elbow method)
            - **Similarity**: Euclidean distance within cluster
            - **Ranking**: Distance-based with normalized confidence
            """)
        else:
            st.markdown("""
            ### Search Details
            - Cluster-based search for efficiency
            - Euclidean distance for similarity
            - Confidence calculated from max distance
            - Results ranked by confidence
            """)

elif mode == "Cluster Analysis":
    st.subheader("Dataset Clustering Visualization")
    st.markdown("Explore how 100,000 images cluster in both 2D and 3D space")
    
    # Create tabs for 2D and 3D views
    tab2d, tab3d = st.tabs(["2D Projection", "3D Projection"])
    
    with tab2d:
        st.markdown("### 2D Cluster Visualization (PCA Projection)")
        
        with st.spinner("Generating 2D projection..."):
            data, labels = engine.get_clusters()
        
        if data is not None and labels is not None:
            df = pd.DataFrame(data, columns=['x', 'y'])
            df['Cluster'] = labels.astype(str)
            
            # Create scatter plot
            fig = px.scatter(
                df, x='x', y='y', color='Cluster',
                title="K-Means Clustering (2D PCA Projection)",
                template="plotly_dark",
                hover_data={'x': False, 'y': False},
                opacity=0.7
            )
            
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=700,
                showlegend=False
            )
            
            fig.update_traces(marker=dict(size=4))
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### 2D Projection Details
            
            - **Dimensionality Reduction**: PCA to 2D
            - **Data Points**: 100,000 images
            - **Colors**: Different clusters
            - **Interaction**: Hover to see position, Click-drag to zoom
            - **Purpose**: See global cluster distribution
            """)
        else:
            st.warning("Could not generate 2D cluster visualization.")
    
    with tab3d:
        st.markdown("### 3D Cluster Visualization (Interactive)")
        
        with st.spinner("Generating 3D projection..."):
            data_3d, labels_3d = engine.get_clusters_3d()
        
        if data_3d is not None and labels_3d is not None:
            df_3d = pd.DataFrame(data_3d, columns=['x', 'y', 'z'])
            df_3d['Cluster'] = labels_3d.astype(str)
            
            # Create 3D scatter plot with Plotly
            fig_3d = px.scatter_3d(
                df_3d, x='x', y='y', z='z', color='Cluster',
                title="K-Means Clustering (3D PCA Projection)",
                template="plotly_dark",
                hover_data={'x': False, 'y': False, 'z': False},
                opacity=0.7
            )
            
            fig_3d.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=800,
                showlegend=False,
                scene=dict(
                    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='gray'),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='gray'),
                    zaxis=dict(showgrid=True, gridwidth=1, gridcolor='gray'),
                    bgcolor='rgba(0,0,0,0)'
                )
            )
            
            fig_3d.update_traces(marker=dict(size=3))
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            st.markdown("""
            ### 3D Projection Details
            
            - **Dimensionality Reduction**: PCA to 3D
            - **Data Points**: 100,000 images
            - **Colors**: Different clusters
            - **Interaction**: 
              - Left-click + drag to rotate
              - Right-click + drag to pan
              - Scroll to zoom in/out
              - Double-click to reset view
            - **Purpose**: See deeper cluster structure and relationships
            """)
        else:
            st.warning("Could not generate 3D cluster visualization.")
    
    # Common statistics section
    st.markdown("---")
    st.markdown("### Clustering Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Images", "100,000")
    
    with col2:
        st.metric("Number of Clusters", "300")
    
    with col3:
        st.metric("Avg Images/Cluster", "~333")
    
    with col4:
        st.metric("Feature Dimension", "256D (PCA)")
    
    st.markdown("""
    ### Visualization Information
    
    **Data:**
    - Dataset: Tiny ImageNet (100,000 images, 200 classes)
    - Features: ResNet50 (2048D) → PCA (256D)
    - Clustering: K-Means with K=300
    
    **Why Two Views?**
    - **2D View**: Quick overview, easier to spot macro patterns
    - **3D View**: More dimensional information, better cluster separation visibility
    
    **Interpretation:**
    - Points close together = visually similar images
    - Different colors = different clusters
    - Dense regions = common object categories
    - Sparse regions = unique or rare object types
    """)