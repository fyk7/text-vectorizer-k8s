import numpy as np
import torch
from transformers import BertJapaneseTokenizer, BertModel

# ~/.cache/torch/hub/checkpoints/ にモデルがダウンロードされるが、
# ディスクを圧迫するため削除した方が良さそう
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
MAX_LENGTH = 256
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"


class TextVectorizer(object):
    tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
    model = BertModel.from_pretrained(MODEL_NAME)
    model = model.to(DEVICE)

    def __init__(self):
        pass

    @staticmethod
    def vectorize(text: str) -> np.array:
        # TODO tokenizerやmodelなど大きなオブジェクトを効率よく使い回す手法を調査
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
    def calc_similarity(vect1: np.array, vect2: np.array) -> np.array:
        norm = np.linalg.norm(np.concatenate([vect1, vect2]))
        vect1 = vect1 / norm
        vect2 = vect2 / norm
        return vect1.dot(vect2.T)


    @staticmethod
    def calc_similarity_multi(sent_vectors: np.array) -> np.array:
        norm = np.linalg.norm(sent_vectors, axis=1, keepdims=True)
        sent_vectors_normalized = sent_vectors / norm
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
