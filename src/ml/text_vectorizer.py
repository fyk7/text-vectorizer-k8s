import typing as t

import numpy as np
import pandas as pd
import torch
from transformers import BertJapaneseTokenizer, BertModel


MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
MAX_LENGTH = 256
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"


class TextVectorizer(object):
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)

    @staticmethod
    def vectorize(text: str) -> np.ndarray:
        """
        Method for vectorizing text.
        Parameters
        ----------
        text : str
             Text to be vectorized.
        """
        if not isinstance(text, str):
            raise TypeError(f"Args should be str type.")
        if len(text) >= MAX_LENGTH:
            raise ValueError(f"Text length should be below {MAX_LENGTH}")
        encoding = TextVectorizer.tokenizer(
            text, 
            max_length=MAX_LENGTH, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        encoding = { k: v.to(DEVICE) for k, v in encoding.items() } 
        attention_mask = encoding['attention_mask']

        with torch.no_grad():
            output = TextVectorizer.model(**encoding)
            last_hidden_state = output.last_hidden_state 
            averaged_hidden_state = \
                (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) \
                / attention_mask.sum(1, keepdim=True) 

        return averaged_hidden_state[0].to("cpu").numpy()

    @staticmethod
    def calc_similarity(
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
    def calc_similarity_multi(
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


if __name__ == "__main__":
    tennis_vector = TextVectorizer.vectorize("週末は映画を見に行こう")
    soccer_vector = TextVectorizer.vectorize("野球の試合を見よう")
    dentist_vector = TextVectorizer.vectorize("歯医者に行かなければ")
    print(TextVectorizer.calc_similarity(tennis_vector, soccer_vector))
    print(TextVectorizer.calc_similarity(tennis_vector, dentist_vector))
