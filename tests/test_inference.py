import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from sklearn.exceptions import InconsistentVersionWarning
import torch

from src.inference import ArtifactError, InferenceEngine
from src.verify_inference import main as verify
from tests.helpers import make_index


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = make_index(self.root)

    def test_ranks_only_predicted_cluster_with_cluster_relative_score(self):
        query = self.engine.embeddings[0].copy()
        cluster = self.engine.kmeans_model.predict(query[None])[0]
        expected = np.flatnonzero(self.engine.cluster_labels == cluster)
        distances = np.linalg.norm(self.engine.embeddings[expected] - query, axis=1)
        order = np.argsort(distances, kind="stable")
        with patch.object(self.engine, "extract_features", return_value=query):
            results = self.engine.search(None, k=100)
        self.assertEqual([r["filename"] for r in results],
                         [Path(self.engine.filenames[i]).name for i in expected[order]])
        np.testing.assert_allclose([r["distance"] for r in results], distances[order])
        np.testing.assert_allclose([r["score"] for r in results], 1 - distances[order] / distances.max())
        self.assertTrue(all(r["cluster"] == cluster for r in results))
        self.assertEqual([r["rank"] for r in results], list(range(1, len(results) + 1)))

    def test_scores_use_entire_cluster_even_when_top_k_is_one(self):
        query = self.engine.embeddings[0] + 0.1
        with patch.object(self.engine, "extract_features", return_value=query):
            one = self.engine.search(None, k=1)
            all_results = self.engine.search(None, k=100)
        self.assertEqual(one, all_results[:1])

    def test_identical_embeddings_have_finite_scores_and_stable_ties(self):
        self.engine.embeddings = np.zeros((8, 3), dtype=np.float32)
        self.engine.cluster_labels = np.zeros(8, dtype=int)
        with patch.object(self.engine, "extract_features", return_value=np.zeros(3)), \
                patch.object(self.engine.kmeans_model, "predict", return_value=np.array([0])):
            results = self.engine.search(None, k=3)
        self.assertEqual([r["filename"] for r in results], ["0.JPEG", "1.JPEG", "2.JPEG"])
        self.assertEqual([r["score"] for r in results], [1.0] * 3)

    def test_empty_predicted_cluster_returns_no_results(self):
        with patch.object(self.engine, "extract_features", return_value=np.zeros(3)), \
                patch.object(self.engine.kmeans_model, "predict", return_value=np.array([99])):
            self.assertEqual(self.engine.search(None), [])

    def test_invalid_k_rejected_before_inference(self):
        for k in [0, -1, 1.5, True, "5", None]:
            with self.subTest(k=k), self.assertRaisesRegex(ValueError, "positive integer"):
                self.engine.search(None, k=k)

    def test_missing_corpus_fails_before_weight_download(self):
        self.engine.image_path(self.engine.filenames[0]).unlink()
        with patch.object(self.engine, "_init_model") as init:
            with self.assertRaisesRegex(FileNotFoundError, "corpus images are missing"):
                self.engine.search(Image.new("RGB", (20, 20)))
            init.assert_not_called()

    def test_disappearing_result_image_is_not_reported_as_success(self):
        self.engine.check_corpus()
        self.engine.image_path(self.engine.filenames[0]).unlink()
        with patch.object(self.engine, "extract_features", return_value=self.engine.embeddings[0]):
            with self.assertRaisesRegex(FileNotFoundError, "disappeared"):
                self.engine.search(None)

    def test_required_files_and_corrupt_pickle_fail_clearly(self):
        for name in ["embeddings.npy", "cluster_labels.npy", "filenames.json", "pca_model.pkl", "kmeans_model.pkl"]:
            path = self.root / name
            original = path.read_bytes()
            path.unlink()
            with self.subTest(name=name), self.assertRaises(ArtifactError):
                InferenceEngine(self.root)
            path.write_bytes(original)
        (self.root / "pca_model.pkl").write_bytes(b"invalid pickle")
        with self.assertRaises(ArtifactError):
            InferenceEngine(self.root)

    def test_invalid_artifact_arrays_are_rejected(self):
        original = self.engine.embeddings.copy()
        for embeddings in [original[:-1], original[:, :2], np.full_like(original, np.nan), original.astype(int)]:
            np.save(self.root / "embeddings.npy", embeddings)
            with self.subTest(shape=embeddings.shape), self.assertRaises(ArtifactError):
                InferenceEngine(self.root)
        np.save(self.root / "embeddings.npy", original)
        for labels in [self.engine.cluster_labels[:-1], np.full(8, 50), self.engine.cluster_labels.astype(float),
                       1 - self.engine.cluster_labels]:
            np.save(self.root / "cluster_labels.npy", labels)
            with self.subTest(labels=labels.tolist()), self.assertRaises(ArtifactError):
                InferenceEngine(self.root)

    def test_unsafe_duplicate_and_mismatched_filenames_rejected(self):
        for names in [self.engine.filenames[:-1], ["same.JPEG"] * 8,
                      ["../outside.jpg"] + self.engine.filenames[1:],
                      ["C:/outside.jpg"] + self.engine.filenames[1:],
                      ["/outside.jpg"] + self.engine.filenames[1:],
                      ["."] + self.engine.filenames[1:],
                      ["train//file.JPEG"] + self.engine.filenames[1:],
                      [None] + self.engine.filenames[1:]]:
            (self.root / "filenames.json").write_text(json.dumps(names), encoding="utf-8")
            with self.subTest(names=names[:1]), self.assertRaises(ArtifactError):
                InferenceEngine(self.root)

    def test_incompatible_model_dimensions_rejected(self):
        self.engine.pca_model.n_components_ = 99
        with (self.root / "pca_model.pkl").open("wb") as stream:
            pickle.dump(self.engine.pca_model, stream)
        with self.assertRaisesRegex(ArtifactError, "PCA dimensions"):
            InferenceEngine(self.root)

    def test_model_version_mismatch_is_an_error(self):
        warning = InconsistentVersionWarning(estimator_name="PCA", current_sklearn_version="new",
                                             original_sklearn_version="old")
        with patch("src.inference.pickle.load", side_effect=warning):
            with self.assertRaises(ArtifactError):
                InferenceEngine(self.root)

    def test_projection_shapes_and_cache(self):
        for project, dimensions in [(self.engine.get_clusters, 2), (self.engine.get_clusters_3d, 3)]:
            data, labels = project()
            self.assertEqual(data.shape, (8, dimensions))
            np.testing.assert_array_equal(labels, self.engine.cluster_labels)
            with patch("src.inference.PCA", side_effect=AssertionError("unexpected refit")):
                self.assertIs(project()[0], data)

    def test_model_initialization_failure_is_explicit(self):
        with patch("src.inference.models.resnet50", side_effect=OSError("offline")):
            with self.assertRaisesRegex(RuntimeError, "Cannot initialize ResNet50"):
                self.engine.extract_features(Image.new("RGB", (20, 20)))

    def test_feature_extraction_uses_pca_and_rgb_conversion(self):
        class ConstantFeatures(torch.nn.Module):
            def forward(self, tensor):
                return torch.ones((tensor.shape[0], 2048, 1, 1), device=tensor.device)
        self.engine.model = ConstantFeatures().to(self.engine.device)
        expected = self.engine.pca_model.transform(np.ones((1, 2048), dtype=np.float32))[0]
        actual = self.engine.extract_features(Image.new("L", (40, 60)))
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    def test_feature_failure_is_not_a_zero_vector(self):
        self.engine.model = torch.nn.Identity()
        with self.assertRaisesRegex(RuntimeError, "Feature extraction failed"):
            self.engine.extract_features(Image.new("RGB", (20, 20)))

    def test_initialization_is_lazy(self):
        with patch("src.inference.models.resnet50", side_effect=AssertionError("download")):
            engine = InferenceEngine(self.root)
        self.assertIsNone(engine.model)

    def test_smoke_check_returns_nonzero_on_failure(self):
        with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(verify(["--data-dir", str(self.root), "--artifacts-only"]), 0)
            self.engine.image_path(self.engine.filenames[0]).unlink()
            self.assertEqual(verify(["--data-dir", str(self.root)]), 1)


if __name__ == "__main__":
    unittest.main()
