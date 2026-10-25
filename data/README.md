# Index Artifacts and Image Corpus

The application reads five supplied files from its artifact directory.

## Supplied files

| File | Documented role and dimensions |
|---|---|
| `embeddings.npy` | PCA-reduced image features, documented as `(100000, 256)`; file size 102,400,128 bytes including the NumPy header |
| `cluster_labels.npy` | Cluster assignments corresponding to the embedding rows; documented as 100,000 labels across 300 clusters |
| `filenames.json` | Image paths corresponding to embedding rows |
| `pca_model.pkl` | Saved scikit-learn PCA model; documented reduction from 2048 to 256 dimensions |
| `kmeans_model.pkl` | Saved scikit-learn K-means model; documented configuration of 300 clusters |

The application uses **PyTorch and torchvision ResNet50** for image feature extraction. The supplied `requirements.txt` lists dependencies without version pins. Earlier recorded package-version notes do not establish compatibility with the supplied models. Only load trusted pickle files, since deserialization can execute code.

## Corpus and local files

The index refers to the Tiny ImageNet training image collection. The repository's existing corpus layout is:

```text
data/train/<class>/images/<filename>.JPEG
```

During search, the source joins each saved filename to the artifact directory. If that image path does not exist, it also tries the same filename under a `tiny-imagenet-200/` subdirectory. These lookups depend on the paths stored in `filenames.json`.

The local `tiny-imagenet-200.zip` archive and extracted image directories are excluded by `.gitignore`. The [Tiny ImageNet download](https://cs231n.stanford.edu/tiny-imagenet-200.zip) is referenced by the project; dataset attribution.

## Current source interfaces

`InferenceEngine` accepts `data_dir`, with a default of `./data` relative to the current working directory. The Streamlit app uses that default. Image paths are resolved from the same directory as described above.

The supplied source does not expose an `image_dir` argument or `check_corpus()` method. Likewise, `src/verify_inference.py` has a `verify()` function using the default engine; it does not parse `--data-dir` or `--image-dir` arguments.

## Notebook export locations

| Notebook | Export statements present in the supplied source |
|---|---|
| [retrieval.ipynb](../notebooks/retrieval.ipynb) | Writes all five index filenames using relative paths |
| [clustering.ipynb](../notebooks/clustering.ipynb) | Writes embeddings, filenames, PCA, and K-means files using relative paths; uses a different experimental configuration |
| [colab_training.ipynb](../notebooks/colab_training.ipynb) | STL10 experiment exporting embeddings, filenames, and a K-means model using relative paths |
