import os
import nltk
import torch
import nltk.data
import numpy as np
from scipy import spatial
from cog import BasePredictor, Input, BaseModel
from sentence_transformers import SentenceTransformer

os.environ['NLTK_DATA'] = 'models/nltk_data'


class Output(BaseModel):
    intervals: list
    adjacency: list


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = SentenceTransformer(
            'all-mpnet-base-v2', cache_folder='models'
        )
        self.sent_detector = nltk.data.load(
            'tokenizers/punkt/english.pickle', )

    def predict(
        self,
        text: str = Input(description="Text to summarize"),
    ) -> Output:
        """Run a single prediction on the model"""
        sentence_ranges = list(self.sent_detector.span_tokenize(text))
        sentences = [text[start:end] for start, end in sentence_ranges]
        vectors = [self.model.encode(s) for s in sentences]
        adjacency = normalized_adjacency(
            torch.stack([
                torch.tensor(
                    [max(0, 1 - float(spatial.distance.cosine(a, b))) for a in vectors])
                for b in vectors
            ]).fill_diagonal_(0.)
        )

        return Output(
            intervals=sentence_ranges,
            adjacency=np.nan_to_num(adjacency.numpy()).tolist(),
        )


def normalized_adjacency(A):
    normalized_D = degree_power(A, -0.5)
    return torch.from_numpy(normalized_D.dot(A).dot(normalized_D))


def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).ravel()
    D = np.diag(degrees)
    return D
