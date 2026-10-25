"""
Inference Engine for Image Retrieval
Matches the MLProject (2).ipynb Colab notebook implementation
- K-Means clustering (K=300)
- Cluster-based search using Euclidean distance
- Confidence scores: 1.0 - (distance / max_distance)
"""

import torch
import numpy as np
import os
import json
import pickle
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from torchvision import models
from src.utils import get_transform


class InferenceEngine:
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.embeddings = None
        self.filenames = None
        self.cluster_labels = None
        self.kmeans_model = None
        self.pca_model = None
        self.model = None
        self.transform = None
        self.device = None
        self.is_mock = False
        
        self._load_artifacts()
        self._init_model()

    def _load_artifacts(self):
        """
        Loads all artifacts from the data directory.
        Matches Colab notebook's artifact loading.
        """
        embeddings_path = os.path.join(self.data_dir, 'embeddings.npy')
        filenames_path = os.path.join(self.data_dir, 'filenames.json')
        cluster_labels_path = os.path.join(self.data_dir, 'cluster_labels.npy')
        pca_path = os.path.join(self.data_dir, 'pca_model.pkl')
        kmeans_path = os.path.join(self.data_dir, 'kmeans_model.pkl')
        
        try:
            # Load embeddings (PCA-reduced to 256D)
            if os.path.exists(embeddings_path):
                print(f"Loading embeddings from {embeddings_path}...")
                self.embeddings = np.load(embeddings_path)
                print(f"  Shape: {self.embeddings.shape}")
            else:
                raise FileNotFoundError("embeddings.npy not found")

            # Load filenames
            if os.path.exists(filenames_path):
                print(f"Loading filenames from {filenames_path}...")
                with open(filenames_path, 'r') as f:
                    self.filenames = json.load(f)
                print(f"  Count: {len(self.filenames)}")
            else:
                raise FileNotFoundError("filenames.json not found")

            # Load cluster labels
            if os.path.exists(cluster_labels_path):
                print(f"Loading cluster labels from {cluster_labels_path}...")
                self.cluster_labels = np.load(cluster_labels_path)
                print(f"  Shape: {self.cluster_labels.shape}")
                print(f"  Unique clusters: {len(np.unique(self.cluster_labels))}")
            else:
                raise FileNotFoundError("cluster_labels.npy not found")

            # Load PCA model
            if os.path.exists(pca_path):
                print(f"Loading PCA model from {pca_path}...")
                with open(pca_path, 'rb') as f:
                    self.pca_model = pickle.load(f)
                print(f"  Input dims: {self.pca_model.n_features_in_} → Output: {self.pca_model.n_components_}")
            else:
                print("PCA model not found.")

            # Load KMeans model
            if os.path.exists(kmeans_path):
                print(f"Loading KMeans model from {kmeans_path}...")
                with open(kmeans_path, 'rb') as f:
                    self.kmeans_model = pickle.load(f)
                print(f"  Number of clusters (K): {self.kmeans_model.n_clusters}")
            else:
                raise FileNotFoundError("kmeans_model.pkl not found")

            print("All artifacts loaded successfully!")

        except Exception as e:
            print(f"Error loading artifacts: {e}")
            self.is_mock = True

    def _init_model(self):
        """
        Initializes the ResNet50 model for feature extraction.
        Matches Colab notebook's model initialization.
        """
        if self.is_mock:
            return

        try:
            print("Initializing ResNet50 model...")
            
            # Load pre-trained ResNet50
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.model = models.resnet50(weights=weights)
            
            # Remove the classification layer (fc) to get 2048D features
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.transform = get_transform()
            
            print(f"ResNet50 model initialized on {self.device}")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            self.is_mock = True

    def extract_features(self, image):
        """
        Extracts 2048D features from image and applies PCA reduction to 256D.
        Matches Colab notebook's feature extraction pipeline.
        """
        if self.model is None or self.transform is None:
            return np.random.rand(256)

        try:
            # Transform and move to device
            img_t = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract 2048D features from ResNet50
            with torch.no_grad():
                features_2048 = self.model(img_t)
            
            # Flatten to 1D
            features_2048 = features_2048.cpu().numpy().flatten()
            
            # Apply PCA reduction to 256D (matching Colab notebook)
            if self.pca_model is not None:
                features_256d = self.pca_model.transform(features_2048.reshape(1, -1)).flatten()
            else:
                # Fallback if PCA model not available
                features_256d = features_2048[:256] if len(features_2048) >= 256 else features_2048
            
            return features_256d
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(256)

    def search(self, query_image, k=5):
        """
        Searches for similar images using cluster-based Euclidean distance.
        Exactly matches the Colab notebook's find_similar_images function.
        
        Algorithm:
        1. Extract query image features (2048D → 256D via PCA)
        2. Predict cluster using KMeans
        3. Find all images in that cluster
        4. Calculate Euclidean distances to all images in cluster
        5. Sort by distance (ascending)
        6. Calculate confidence scores: 1.0 - (distance / max_distance)
        7. Return top k results
        """
        if self.is_mock:
            return self._mock_search(k)
        
        try:
            # 1. Extract query image features
            query_features = self.extract_features(query_image)
            
            # 2. Predict the cluster for the query image
            query_features_2d = query_features.reshape(1, -1)
            query_cluster_id = self.kmeans_model.predict(query_features_2d)[0]
            
            print(f"Query cluster ID: {query_cluster_id}")
            
            # 3. Find all images belonging to this cluster
            indices_in_cluster = np.where(self.cluster_labels == query_cluster_id)[0]
            
            print(f"Images in cluster: {len(indices_in_cluster)}")
            
            # 4. Calculate Euclidean distances to all images in the cluster
            distances = []
            for idx in indices_in_cluster:
                img_embedding = self.embeddings[idx]
                dist = euclidean(query_features, img_embedding)
                distances.append((dist, idx))
            
            # 5. Sort by distance (ascending - smaller distances first)
            distances.sort(key=lambda x: x[0])
            
            # 6. Calculate confidence scores and build results
            results = []
            
            if len(distances) > 0:
                # Get distances for normalization
                all_dists = [d[0] for d in distances]
                max_dist = max(all_dists) if all_dists else 1.0
                
                print(f"Max distance in cluster: {max_dist:.4f}")
                
                # 7. Get top k results
                for i, (dist, idx) in enumerate(distances[:k]):
                    filename = self.filenames[idx]
                    
                    # Construct full path - handle both formats
                    full_path = os.path.join(self.data_dir, filename)
                    
                    # Verify path exists
                    if not os.path.exists(full_path):
                        # Try alternative path format
                        alt_path = os.path.join(self.data_dir, 'tiny-imagenet-200', filename)
                        if os.path.exists(alt_path):
                            full_path = alt_path
                    
                    # Calculate confidence score: 1.0 - (distance / max_distance)
                    # This gives 1.0 for closest match and 0.0 for farthest in cluster
                    if max_dist > 0:
                        confidence = 1.0 - (dist / max_dist)
                    else:
                        confidence = 1.0
                    
                    # Ensure confidence is in [0, 1] range
                    confidence = max(0.0, min(1.0, confidence))
                    
                    results.append({
                        'path': full_path,
                        'filename': os.path.basename(full_path),
                        'confidence': float(confidence),
                        'distance': float(dist),
                        'cluster': int(query_cluster_id),
                        'rank': i + 1
                    })
                
                print(f"Found {len(results)} results with confidence scores")
            else:
                print("No images found in cluster")
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _mock_search(self, k):
        """
        Returns mock results for testing when artifacts are not available.
        """
        return [
            {
                'path': 'https://via.placeholder.com/150',
                'filename': f'image_{i}.jpg',
                'confidence': 1.0 - (i * 0.1),
                'distance': i * 0.1,
                'cluster': 0,
                'rank': i + 1
            }
            for i in range(k)
        ]

    def get_clusters(self):
        """
        Returns 2D projection of embeddings for cluster visualization.
        """
        if self.is_mock:
            return self._mock_clusters()
            
        try:
            if self.embeddings is None or self.cluster_labels is None:
                return None, None
            
            print("Reducing dimensionality with PCA for visualization...")
            pca_2d = PCA(n_components=2)
            data_2d = pca_2d.fit_transform(self.embeddings)
            
            print(f"2D projection shape: {data_2d.shape}")
            print(f"Explained variance: {np.sum(pca_2d.explained_variance_ratio_):.4f}")
            
            return data_2d, self.cluster_labels
            
        except Exception as e:
            print(f"Clustering error: {e}")
            return None, None

    def get_clusters_3d(self):
        """
        Returns 3D projection of embeddings for 3D cluster visualization.
        """
        if self.is_mock:
            return self._mock_clusters_3d()
            
        try:
            if self.embeddings is None or self.cluster_labels is None:
                return None, None
            
            print("Reducing dimensionality with PCA for 3D visualization...")
            pca_3d = PCA(n_components=3)
            data_3d = pca_3d.fit_transform(self.embeddings)
            
            print(f"3D projection shape: {data_3d.shape}")
            print(f"Explained variance: {np.sum(pca_3d.explained_variance_ratio_):.4f}")
            
            return data_3d, self.cluster_labels
            
        except Exception as e:
            print(f"3D Clustering error: {e}")
            return None, None

    def _mock_clusters(self):
        """
        Returns mock cluster visualization data (2D).
        """
        n_points = 50
        c1 = np.random.normal(loc=[1, 1], scale=0.5, size=(n_points, 2))
        c2 = np.random.normal(loc=[5, 5], scale=0.5, size=(n_points, 2))
        c3 = np.random.normal(loc=[1, 5], scale=0.5, size=(n_points, 2))
        
        data = np.vstack([c1, c2, c3])
        labels = np.array([0]*n_points + [1]*n_points + [2]*n_points)
        return data, labels

    def _mock_clusters_3d(self):
        """
        Returns mock cluster visualization data (3D).
        """
        n_points = 50
        c1 = np.random.normal(loc=[1, 1, 1], scale=0.5, size=(n_points, 3))
        c2 = np.random.normal(loc=[5, 5, 5], scale=0.5, size=(n_points, 3))
        c3 = np.random.normal(loc=[1, 5, 1], scale=0.5, size=(n_points, 3))
        
        data = np.vstack([c1, c2, c3])
        labels = np.array([0]*n_points + [1]*n_points + [2]*n_points)
        return data, labels
