import sys
import os
import numpy as np
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import InferenceEngine

def verify():
    print("Initializing InferenceEngine...")
    engine = InferenceEngine()
    
    if engine.is_mock:
        print("FAIL: Engine is in mock mode.")
        return
    
    print("PASS: Engine initialized in real mode.")
    
    # Test Search
    print("\nTesting Search...")
    # Create a dummy image
    dummy_img = Image.new('RGB', (224, 224), color = 'red')
    results = engine.search(dummy_img, k=3)
    
    if len(results) == 3:
        print(f"PASS: Search returned {len(results)} results.")
        print("Top result:", results[0])
    else:
        print(f"FAIL: Search returned {len(results)} results.")
        
    # Test Clustering
    print("\nTesting Clustering...")
    data, labels = engine.get_clusters()
    
    if data is not None and labels is not None:
        print(f"PASS: Clustering returned data shape {data.shape} and labels shape {labels.shape}")
    else:
        print("FAIL: Clustering returned None.")

if __name__ == "__main__":
    verify()
