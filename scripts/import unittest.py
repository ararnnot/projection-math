import unittest
import numpy as np
from scripts.compare_universes import extend_projection

# scripts/test_compare_universes.py


class TestExtendProjection(unittest.TestCase):
    def setUp(self):
        self.universe_1 = ["this", "is an", "example"]
        self.universe_2 = ["hello", "world"]
        self.projection1 = [0.1, 0.2, 0.3]
        self.word_embedding_path = "resources/glove.6B.50d.kv"

        self.expected = {
            "extension_S41": np.array([0.19830016, 0.1938949]),
            "extension_S42": np.array([0, 0]),
            "extension_S43": np.array([0.12476815, 0.1])
        }

    def test_extend_projection(self):
        result = extend_projection(
            universe1=self.universe_1,
            universe2=self.universe_2,
            projection1=self.projection1,
            extend_methods=[1, 2, 3],
            word_embedding_path=self.word_embedding_path
        )
        for key in self.expected:
            self.assertIn(key, result)
            np.testing.assert_allclose(result[key], self.expected[key], rtol=1e-6, atol=1e-6)

if __name__ == "__main__":
    unittest.main()