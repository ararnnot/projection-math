
import numpy as np
import pandas as pd
from typing import (
    Sequence,
    Union
)

# Recall: pip install git+https://github.com/ararnnot/projection-math.git
from projection_math.compare_universes.methods import Compare_Universes, extract_s4_number

def universe_comparison(
    universe1: Union[Sequence[str], pd.Series, np.ndarray],
    universe2: Union[Sequence[str], pd.Series, np.ndarray],
    projection1: Union[Sequence[float], pd.Series, np.ndarray],
    projection2: Union[Sequence[float], pd.Series, np.ndarray],
    extend_methods: Union[int, str, list] = [1, 2, 3],
    extend_universe: Union[int, list] = 2,
    word_embedding_path: str = None,
    distance: str = 'cosine_diss'
) -> dict[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare two universes based on their projections.
    
    Arguments:
        universe1 (Union[Sequence[str], pd.Series, np.ndarray]): First universe.
        universe2 (Union[Sequence[str], pd.Series, np.ndarray]): Second universe.
        projection1 (Union[Sequence[float], pd.Series, np.ndarray]): Projection of the first universe.
        projection2 (Union[Sequence[float], pd.Series, np.ndarray]): Projection of the second universe.
        word_embedding_path (str): Path to the word embedding model.
        distance (str): Distance metric to use ('cosine_diss' or 'euclidean').
        extend_methods (Union[int, list]): Methods for extension of projection to universe2.
            1: Extension method S4.1
            2: Extension method S4.2
            3: Extension method S4.3
        extend_universe (Union[int, list]): Universe to extend the projection to.
        
    Returns:
        dict: Dictionary containing the comparison results.
    """
    
    if not isinstance(extend_methods, list): extend_methods = [extend_methods]
    for i in range(len(extend_methods)):
        if isinstance(extend_methods[i], str):
            extend_methods[i] = extract_s4_number(extend_methods[i])
            
    if not isinstance(extend_universe, list): extend_universe = [extend_universe]
    
    comparison = Compare_Universes(
        universe1 = universe1,
        universe2 = universe2,
        projection1 = projection1,
        projection2 = projection2,
        word_embedding = word_embedding_path,
        distance = distance
    )
    
    result = {}
    
    if 1 in extend_universe:
        if 1 in extend_methods:
            result["extension_1_S41"], result["error_1_S41"] = comparison.extension_41(ext_universe = 1)
        if 2 in extend_methods:
            result["extension_1_S42"], result["error_1_S42"] = comparison.extension_42(ext_universe = 1)
        if 3 in extend_methods:
            result["extension_1_S43"], result["error_1_S43"] = comparison.extension_43(ext_universe = 1)
        
    if 2 in extend_universe:
        if 1 in extend_methods:
            result["extension_2_S41"], result["error_2_S41"] = comparison.extension_41(ext_universe = 2)
        if 2 in extend_methods:
            result["extension_2_S42"], result["error_2_S42"] = comparison.extension_42(ext_universe = 2)
        if 3 in extend_methods:
            result["extension_2_S43"], result["error_2_S43"] = comparison.extension_43(ext_universe = 2)
        
    return result

def extend_projection(
    universe1: Union[Sequence[str], pd.Series, np.ndarray],
    universe2: Union[Sequence[str], pd.Series, np.ndarray],
    projection1: Union[Sequence[float], pd.Series, np.ndarray],
    extend_methods: Union[int, str, list] = [1, 2, 3],
    word_embedding_path: str = None,
    distance: str = 'cosine_diss'
) -> dict[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extend the projection of universe1 to universe2 (which projection is not given).
    
    Arguments:
        universe1 (Union[Sequence[str], pd.Series, np.ndarray]): First universe.
        universe2 (Union[Sequence[str], pd.Series, np.ndarray]): Second universe.
        projection1 (Union[Sequence[float], pd.Series, np.ndarray]): Projection of the first universe.
        word_embedding_path (str): Path to the word embedding model.
        distance (str): Distance metric to use ('cosine_diss' or 'euclidean').
        extend_methods (Union[int, list]): Methods for extension of projection to universe2.
            1: Extension method S4.1
            2: Extension method S4.2
            3: Extension method S4.3
        
    Returns:
        dict: Dictionary containing the extended projections and errors.
    """
    
    if not isinstance(extend_methods, list): extend_methods = [extend_methods]
    for i in range(len(extend_methods)):
        if isinstance(extend_methods[i], str):
            extend_methods[i] = extract_s4_number(extend_methods[i])
    
    comparison = Compare_Universes(
        universe1 = universe1,
        universe2 = universe2,
        projection1 = projection1,
        projection2 = None,
        word_embedding = word_embedding_path,
        distance = distance
    )
    
    result = {}
    
    if 1 in extend_methods:
        result["extension_S41"], _ = comparison.extension_41(ext_universe = 2)
    if 2 in extend_methods:
        result["extension_S42"], _ = comparison.extension_42(ext_universe = 2)
    if 3 in extend_methods:
        result["extension_S43"], _ = comparison.extension_43(ext_universe = 2)
    
    return result

if __name__ == "__main__":
    
    # Example of comparison
    
    universe_1 = ["term1", "term2", "term3"]
    universe_2 = ["other1", "other2"]
    projection1 = [0.1, 0.2, 0.3]
    projection2 = [0.4, 0.5]
    word_embedding_path = "resources/glove.6B.50d.kv"
    
    result = universe_comparison(
        universe1 = universe_1,
        universe2 = universe_2,
        projection1 = projection1,
        projection2 = projection2,
        extend_methods = [1, 2, 3],
        extend_universe = 2,
        word_embedding_path = word_embedding_path
    )
    
    print("Comparison Result:")
    for key, value in result.items():
        print(f"{key}: {value}")
        
    # Example of extension
    
    result = extend_projection(
        universe1 = universe_1,
        universe2 = universe_2,
        projection1 = projection1,
        extend_methods = [1, 2, 3],
        word_embedding_path = word_embedding_path
    )
    
    print("\nExtension Result:")
    for key, value in result.items():
        print(f"{key}: {value}")
    