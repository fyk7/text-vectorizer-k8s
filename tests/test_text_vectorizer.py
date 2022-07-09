import math
from typing import List

import pytest
import numpy as np
import pandas as pd

from src.ml.text_vectorizer import TextVectorizer


class TestTextVectorizer(object):
    def test_calc_similarity_valid(self):
        input_vector1_np: np.ndarray = np.array([1.0, 2.0, 3.0])
        input_vector2_np: np.ndarray = np.array([2.0, -0.5, 0.0])

        input_vector1_pd: pd.Series = pd.Series([1.0, 2.0, 3.0])
        input_vector2_pd: pd.Series = pd.Series([2.0, -0.5, 0.0])

        input_vector1_list: List[float] = [1.0, 2.0, 3.0]
        input_vector2_list: List[float] = [2.0, -0.5, 0.0]

        expected: float = 0.12964074461289993

        # 近似値を比較する
        assert math.isclose(TextVectorizer.calc_similarity(input_vector1_np, input_vector2_np), expected)
        assert math.isclose(TextVectorizer.calc_similarity(input_vector1_pd, input_vector2_pd), expected)
        assert math.isclose(TextVectorizer.calc_similarity(input_vector1_list, input_vector2_list), expected)

    def test_calc_similarity_raises(self):
        input_vector1_str: List[str] = ["1.0", "2.0", "3.0"] 
        input_vector2_str: List[str] = ["2.0", "-0.5", "0.0"] 

        with pytest.raises(TypeError):
            TextVectorizer.calc_similarity(input_vector1_str, input_vector2_str)

    def test_calc_similarity_multi_valid(self):
        input_vector_np: np.ndarray = np.array([[1.0, 2.0, 3.0], [2.0, -0.5, 0.0], [1.0, 2.0, 3.0]])
        input_vector_pd: pd.DataFrame = pd.DataFrame([[1.0, 2.0, 3.0], [2.0, -0.5, 0.0], [1.0, 2.0, 3.0]])
        input_vector_list: List[float] = [[1.0, 2.0, 3.0], [2.0, -0.5, 0.0], [1.0, 2.0, 3.0]]

        expected: np.ndarray = np.array([2, 0, 0])

        assert (TextVectorizer.calc_similarity_multi(input_vector_np) == expected).all()
        assert (TextVectorizer.calc_similarity_multi(input_vector_pd) == expected).all()
        assert (TextVectorizer.calc_similarity_multi(input_vector_list) == expected).all()

    def test_calc_similarity_multi_raises(self):
        input_vector1_str: List[str] = [["1.0", "2.0", "3.0"], ["2.0", "-0.5", "0.0"]] 
        input_vector2_dim: List[str] = np.array([2.0, -0.5, 0.0]) 

        with pytest.raises(TypeError):
            TextVectorizer.calc_similarity_multi(input_vector1_str)

        with pytest.raises(ValueError):
            TextVectorizer.calc_similarity_multi(input_vector2_dim)
