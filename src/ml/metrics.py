import typing as t
import numpy as np
import pandas as pd


class SimilarityCalculator(object):
    @staticmethod
    def calc_cosine_similarity(
        sent_vector1: t.Union[np.ndarray, pd.Series, t.List[float]],
        sent_vector2: t.Union[np.ndarray, pd.Series, t.List[float]],
        eps: float = 1e-9,
    ) -> float:
        """
        Function to calculate the similarity between two vectorized sentences
        Parameters
        ----------
        sent_vector1, sent_vector2 : t.Union[np.array, pd.Series, t.List[float]]
            Vectorized text.
        eps : float, default=1e-9
            A small number to avoid ZeroDevisionError if the norm is 0 when normalizing a vector.
        """
        
        def __convert_vec2numpy(vect: t.Any) -> np.ndarray:
            if isinstance(vect, np.ndarray):
                return vect
            elif isinstance(vect, pd.Series):
                return vect.values
            elif isinstance(vect, list):
                return np.array(vect)
            else:
                raise TypeError("Args should be in (np.array, pd.Series, List[float])")

        sent_vector1 = __convert_vec2numpy(sent_vector1)
        sent_vector2 = __convert_vec2numpy(sent_vector2)

        norm1 = np.linalg.norm(sent_vector1)
        norm2 = np.linalg.norm(sent_vector2)
        vect1_normarized =  sent_vector1 / (norm1 + eps)
        vect2_normarized = sent_vector2 / (norm2 + eps)

        return vect1_normarized.dot(vect2_normarized.T)

    @staticmethod
    def calc_cosine_similarity_multi(
        sent_vectors: t.Union[np.ndarray, pd.DataFrame, t.List[t.List[float]]],
        eps: float = 1e-9,
    ) -> np.ndarray:
        """
        Returns an array of the indexes of the most similar texts.
        Parameters
        ----------
        sent_vectors : t.Union[np.array, pd.DataFrame, t.List[float]]
            Vectorized text.
        eps : float, default=1e-9
            A small number to avoid ZeroDevisionError if the norm is 0 when normalizing a vector.
        """

        def __convert_vec2numpy_2dim(vect: t.Any) -> np.ndarray:
            if isinstance(vect, np.ndarray):
                tmp_vect = vect
            elif isinstance(vect, pd.DataFrame):
                tmp_vect = vect.values
            elif isinstance(vect, list):
                tmp_vect = np.array(vect)
            else:
                raise TypeError("Args should be in (np.array, pd.Series, List[float])")
            if tmp_vect.ndim != 2:
                raise ValueError("The dimension of the vector should be two-dimensional.")
            return tmp_vect

        sent_vectors = __convert_vec2numpy_2dim(sent_vectors)

        norm = np.linalg.norm(sent_vectors, axis=1, keepdims=True)
        sent_vectors_normalized = sent_vectors / (norm + eps)
        sim_matrix = sent_vectors_normalized.dot(sent_vectors_normalized.T)

        # To avoid outputting the same articles as the input.
        np.fill_diagonal(sim_matrix, -1)
        similars = sim_matrix.argmax(axis=1)
        return similars
