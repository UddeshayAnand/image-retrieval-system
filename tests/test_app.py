from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from tests.helpers import make_index


APP = Path(__file__).resolve().parents[1] / "app" / "main.py"


class AppTests(unittest.TestCase):
    def setUp(self):
        # AppTest executes the script as __main__; clear its cache between fixtures too.
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)

    def test_startup_error_is_visible_without_uncaught_exception(self):
        with patch("src.inference.InferenceEngine", side_effect=ValueError("Missing PCA artifact")):
            app = AppTest.from_file(str(APP)).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn("Missing PCA artifact", app.error[0].value)

    def test_missing_corpus_error_and_cluster_navigation(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = make_index(directory)
            engine.image_path(engine.filenames[0]).unlink()
            with patch("src.inference.InferenceEngine", return_value=engine):
                app = AppTest.from_file(str(APP)).run(timeout=30)
                self.assertFalse(app.exception)
                self.assertIn("corpus images are missing", app.error[0].value)
                app.sidebar.radio[0].set_value("Cluster Analysis").run(timeout=30)
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual([metric.value for metric in app.metric], ["8", "2", "3"])
                app.radio[0].set_value(3).run(timeout=30)
                self.assertFalse(app.exception)
