from pytorch_transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import TensorDataset
import logging
import re
from scipy import spatial
from keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class SimilarityScorerBert:

    def __init__(self):
        # Load pre-trained model (weights)
        self.model = BertModel.from_pretrained('bert-base-uncased').to(DEVICE)
        self.model.eval()
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_sentence_embeddings(self, sentences, max_len=None):
        if max_len is None:
            max_len = max(len(sentence) for sentence in sentences)
        input_ids = [self.tokenizer.encode(sentence, add_special_tokens=True) for sentence in sentences]
        input_masks = [[1] * len(id_list) for id_list in input_ids]
        input_ids = pad_sequences(input_ids, maxlen=max_len, padding='post', dtype="long", truncating='post', value=self.tokenizer.pad_token_id)
        input_masks = pad_sequences(input_masks, maxlen=max_len, padding='post', dtype="long", truncating='post')
        tensor_input = torch.tensor(input_ids).to(DEVICE)
        tensor_mask = torch.tensor(input_masks).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(tensor_input, attention_mask=tensor_mask)

        last_hidden = outputs[0]
        embeddings_mean = torch.mean(last_hidden, 1)

        return embeddings_mean

    def return_cosine_similarity(self, vec1, vec2):
        return 1 - spatial.distance.cosine(vec1, vec2)

    def return_similarity_scores(self, target_sentence, comparison_sentences, batch_size=32):
        max_len = max(len(sentence) for sentence in [target_sentence] + comparison_sentences)
        target_embedding = self.get_sentence_embeddings([target_sentence], max_len=max_len).cpu()

        scores = []

        for pos in range(0, len(comparison_sentences), batch_size):
            batch = comparison_sentences[pos:pos + batch_size]
            embeddings = self.get_sentence_embeddings(batch, max_len=max_len).cpu()

            for embedding in embeddings:
                sim = self.return_cosine_similarity(target_embedding, embedding)
                scores.append(sim)

        return [{'sim': score} for score in scores]


class Pooler(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.pool = torch.nn.MaxPool1d(3)

    def forward(self, hidden_states):
        pass




if __name__ == '__main__':
    sentence = 'This is a test sentence'
    sentence = 'This is '

    sim = SimilarityScorerBert()
    print(sim.return_similarity_scores('James Cook was a very good man and a loving husband.', [
        'James Cook was a very good man and a loving husband.',
        'James Cook was a very nice man and a loving husband.',
        'James Cook was a bad man and a terrible husband.',
        'James Cook was a nice person and a good husband.',
        'The sky is blue today and learning history is important.']))
