"""Small fitted artifacts and real temporary images for offline tests."""

import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.inference import InferenceEngine


def make_index(directory):
    directory = Path(directory)
    raw = np.random.default_rng(10).normal(size=(8, 2048)).astype(np.float32)
    pca = PCA(n_components=3, random_state=42)
    embeddings = pca.fit_transform(raw)
    km = KMeans(n_clusters=2, n_init=2, random_state=42).fit(embeddings)
    names = [f"train/class/images/{i}.JPEG" for i in range(len(raw))]
    np.save(directory / "embeddings.npy", embeddings)
    np.save(directory / "cluster_labels.npy", km.labels_)
    (directory / "filenames.json").write_text(json.dumps(names), encoding="utf-8")
    for name, model in [("pca_model.pkl", pca), ("kmeans_model.pkl", km)]:
        with (directory / name).open("wb") as stream:
            pickle.dump(model, stream)
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 12), "red").save(path)
    return InferenceEngine(directory)
