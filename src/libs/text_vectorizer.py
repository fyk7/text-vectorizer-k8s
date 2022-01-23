import typing as t

import numpy as np
import pandas as pd
import torch
from transformers import BertJapaneseTokenizer, BertModel


MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
MAX_LENGTH = 256
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"


# TODO calc_similarity部分はtokenizer, modelに依存しないため、別のクラス・関数に切り出す
class TextVectorizer(object):
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME).to(DEVICE)

    @staticmethod
    def vectorize(text: str) -> np.ndarray:
        """
        文章をベクトル化する関数
        Parameters
        ----------
        text : str
             ベクトル化する文章
        """
        if not isinstance(text, str):
            raise TypeError(f"引数はstring型にしてください!")
        if len(text) >= MAX_LENGTH:
            raise ValueError(f"文字数は{MAX_LENGTH}文字以下にしてください!")
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
        二つのベクトル化された文章の類似度を算出する関数
        Parameters
        ----------
        sent_vector1, sent_vector2 : t.Union[np.array, pd.Series, t.List[float]]
            文章をベクトル化したもので、構成要素がfloatな一次元のベクトル
        eps : float, default=1e-9
            ベクトルを正規化する際のノルムが0の場合ZeroDevisionErrorを回避するための小さな数
        """
        
        def __convert_vec2numpy(vect: t.Any) -> np.ndarray:
            if isinstance(vect, np.ndarray):
                return vect
            elif isinstance(vect, pd.Series):
                return vect.values
            elif isinstance(vect, list):
                return np.array(vect)
            else:
                raise TypeError("引数は、np.array, pd.Series, List[float]のみ許容されています!")

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
        最も類似している文章のindexの配列を返す
        例) 戻り値が[1, 0, 0]の場合、
        index=0の文章はindex=1の文章と最も類似していることとなる
        index=1の文章はindex=0の文章と最も類似していることとなる
        Parameters
        ----------
        sent_vectors : t.Union[np.array, pd.DataFrame, t.List[float]]
            文章をベクトル化したもののリストで、構成要素がfloatな二次元のベクトル
        eps : float, default=1e-9
            ベクトルを正規化する際のノルムが0の場合ZeroDevisionErrorを回避するための小さな数
        """

        def __convert_vec2numpy_2dim(vect: t.Any) -> np.ndarray:
            if isinstance(vect, np.ndarray):
                tmp_vect = vect
            elif isinstance(vect, pd.DataFrame):
                tmp_vect = vect.values
            elif isinstance(vect, list):
                tmp_vect = np.array(vect)
            else:
                raise TypeError("引数の型は、np.array, pd.Series, List[float]のみ許容されています!")
            if tmp_vect.ndim != 2:
                raise ValueError("calc_similarity_multiではベクトルの次元は二次元にしてください!")
            return tmp_vect

        sent_vectors = __convert_vec2numpy_2dim(sent_vectors)

        norm = np.linalg.norm(sent_vectors, axis=1, keepdims=True)
        sent_vectors_normalized = sent_vectors / (norm + eps)
        sim_matrix = sent_vectors_normalized.dot(sent_vectors_normalized.T)

        # 入力と同じ記事が出力されることを避けるため、
        # 類似度行列の対角要素の値を小さくしておく。
        np.fill_diagonal(sim_matrix, -1)
        similars = sim_matrix.argmax(axis=1)
        return similars


if __name__ == "__main__":
    tennis_vector = TextVectorizer.vectorize("週末はテニスに行こう")
    soccer_vector = TextVectorizer.vectorize("サッカーの試合を見よう")
    dentist_vector = TextVectorizer.vectorize("虫歯が痛い")
    print(TextVectorizer.calc_similarity(tennis_vector, soccer_vector))
    print(TextVectorizer.calc_similarity(tennis_vector, dentist_vector))
