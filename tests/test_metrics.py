import math
from typing import List

import pytest
import numpy as np
import pandas as pd

from src.ml.metrics import SimilarityCalculator


@pytest.mark.parametrize(
    'vector1, vector2',
    [
        (np.array([1.0, 2.0, 3.0]), np.array([2.0, -0.5, 0.0])),
        (pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, -0.5, 0.0])),
        ([1.0, 2.0, 3.0], [2.0, -0.5, 0.0])
    ]
)
def test_calc_similarity_valid(vector1, vector2):
    expected: float = 0.12964074461289993
    assert math.isclose(SimilarityCalculator.calc_cosine_similarity(vector1, vector2), expected)

def test_calc_similarity_raises():
    input_vector1_str: List[str] = ["1.0", "2.0", "3.0"] 
    input_vector2_str: List[str] = ["2.0", "-0.5", "0.0"] 

    with pytest.raises(TypeError):
        SimilarityCalculator.calc_cosine_similarity(input_vector1_str, input_vector2_str)

@pytest.mark.parametrize(
    'vector_list',
    [
        np.array([[1.0, 2.0, 3.0], [2.0, -0.5, 0.0], [1.0, 2.0, 3.0]]),
        pd.DataFrame([[1.0, 2.0, 3.0], [2.0, -0.5, 0.0], [1.0, 2.0, 3.0]]),
        [[1.0, 2.0, 3.0], [2.0, -0.5, 0.0], [1.0, 2.0, 3.0]]
    ]
)
def test_calc_similarity_multi_valid(vector_list):
    expected: np.ndarray = np.array([2, 0, 0])
    assert (SimilarityCalculator.calc_cosine_similarity_multi(vector_list) == expected).all()

def test_calc_similarity_multi_raises():
    input_vector1_str: List[str] = [["1.0", "2.0", "3.0"], ["2.0", "-0.5", "0.0"]] 
    input_vector2_dim: List[str] = np.array([2.0, -0.5, 0.0]) 

    with pytest.raises(TypeError):
        SimilarityCalculator.calc_cosine_similarity_multi(input_vector1_str)

    with pytest.raises(ValueError):
        SimilarityCalculator.calc_cosine_similarity_multi(input_vector2_dim)
