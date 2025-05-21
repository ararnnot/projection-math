import unittest
import numpy as np
from projection_math.compare_universes.methods import Compare_Universes

class TestExtendProjection(unittest.TestCase):
    def setUp(self):
        self.universe1 = ["this", "is an", "example"]
        self.universe2 = ["hello", "world"]
        self.projection1 = [0.1, 0.2, 0.3]
        self.projection2 = [0.4, 0.5]

        self.expected = {
            "extension_1_S41": np.array([0.47228012, 0.46728196, 0.46636635]),
            "extension_1_S42": np.array([0.46164944, 0.45137181, 0.44932117]),
            "extension_1_S43": np.array([0.47838535, 0.47482889, 0.47514636]),
            "error_1_S41": 0.48755502040326704,
            "error_1_S42": 0.46505367258700137,
            "error_1_S43": 0.4993822583584017,
            "extension_2_S41": np.array([0.19830016, 0.1938949]),
            "extension_2_S42": np.array([0.27591535, 0.23050041]),
            "extension_2_S43": np.array([0.12476815, 0.1]),
            "error_2_S41": 0.36658308500700587,
            "error_2_S42": 0.29669349149939717,
            "error_2_S43": 0.48554358335321124,
        }

    def test_comparation(self):
        
        comparison = Compare_Universes(
            universe1 = self.universe1,
            universe2 = self.universe2,
            projection1 = self.projection1,
            projection2 = self.projection2
        )
        
        result = {}
        result["extension_1_S41"], result["error_1_S41"] = comparison.extension_41(ext_universe = 1)
        result["extension_1_S42"], result["error_1_S42"] = comparison.extension_42(ext_universe = 1)
        result["extension_1_S43"], result["error_1_S43"] = comparison.extension_43(ext_universe = 1)
        result["extension_2_S41"], result["error_2_S41"] = comparison.extension_41(ext_universe = 2)
        result["extension_2_S42"], result["error_2_S42"] = comparison.extension_42(ext_universe = 2)
        result["extension_2_S43"], result["error_2_S43"] = comparison.extension_43(ext_universe = 2)

        assert len([r for r in result if r is not None]) == 12

        for key in result:
            np.testing.assert_allclose(result[key], self.expected[key], rtol=1e-6, atol=1e-6)
            
    def test_extension(self):
        
        comparison = Compare_Universes(
            universe1 = self.universe1,
            universe2 = self.universe2,
            projection1 = self.projection1
        )
        
        result = {}
        result["extension_2_S41"], _ = comparison.extension_41()
        result["extension_2_S42"], _ = comparison.extension_42()
        result["extension_2_S43"], _ = comparison.extension_43()

        
        for key in result:
            np.testing.assert_allclose(result[key], self.expected[key], rtol=1e-6, atol=1e-6)
            

if __name__ == "__main__":
    unittest.main()