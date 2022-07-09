import numpy as np
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
