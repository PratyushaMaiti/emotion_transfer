import logging
import math
import torch
import numpy as np
from pytorch_transformers import OpenAIGPTTokenizer, OpenAIGPTModel, OpenAIGPTLMHeadModel
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PerplexityScorer:

    def __init__(self):
        # Load pre-trained model (weights)
        self.model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
        self.model.eval().to(DEVICE)
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')

    def score(self, sentence):
        tokenize_input = self.tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenize_input)]).to(DEVICE)
        loss = self.model(tensor_input, labels=tensor_input)[0]
        return math.exp(loss)

    def return_lm_scores(self, sentences, normalize_scores=True):
        try:
            scores = [self.score(sentence) for sentence in sentences]
        except RuntimeError:
            logging.warning('Could not get lm scores, set all to 0')
            return [{'lm':0}] * len(sentences)

        if normalize_scores:
            max_score = max(scores)
            min_score = min(scores)
            norm = lambda x: (x-max_score) / (min_score-max_score)

            return [{'lm': score} for score in map(norm, scores)]

        return [{'lm': score} for score in scores]

if __name__ == '__main__':
    a = [
        "i wrote a book, i wrote a book, i wrote a book, i wrote a book,i wrote a book, i wrote a book.",
        "i wrote a book.",
        "i wrote a book about the life of two young people who fall in love with each other."
        ]
    
    ps = PerplexityScorer()

    print(ps.return_lm_scores(a))