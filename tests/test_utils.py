from io import BytesIO
import unittest

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.utils import get_transform, load_image


class ImageTests(unittest.TestCase):
    def test_decode_grayscale_to_rgb_and_release_stream(self):
        stream = BytesIO()
        Image.new("L", (17, 23), 100).save(stream, format="PNG")
        stream.seek(0)
        image = load_image(stream)
        stream.close()
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (100, 100, 100))

    def test_invalid_upload_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "valid JPEG or PNG"):
            load_image(BytesIO(b"not an image"))

    def test_preprocessing_matches_canonical_notebook_on_non_square_image(self):
        pixels = np.random.default_rng(0).integers(0, 256, (100, 300, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        reference = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])(image)
        actual = get_transform()(image)
        self.assertEqual(actual.shape, (3, 224, 224))
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
