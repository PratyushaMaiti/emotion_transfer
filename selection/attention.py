import os

import click
import dill
import logging
import torch
import tabulate

from cfgparser import global_config
from preprocessing import SpacyPreprocessor
from objective.emotion import EmotionClassifier
from objective.emocl.ued_handler import UEDLoader, UEDDataset
from substitution.wordnet import WNSubstitutor
from selection.selection import Selection, Selector
import torch

logger = logging.getLogger(__name__)


class AttentionSelector(Selector):

    def __init__(self, input_sequence, emocl):

        super(AttentionSelector, self).__init__(input_sequence)

        if isinstance(emocl, EmotionClassifier):
            self.emocl = emocl
        else:
            raise TypeError('Argument muste be an EmotionClassifier instance.')

    def get_attention_score(self):
        input = [self.input_tokenized]

        # preprocess/numericalize the input sequence
        input_numericalized = self.emocl.return_input_batch(input, tokenized=True)

        # get attention
        _, attention = self.emocl.get_outputs(input_numericalized)

        return attention

    @classmethod
    def return_attention_table(clf, sentence_tokenized, attentions):
        if isinstance(attentions, torch.Tensor):
            attentions = attentions.squeeze().tolist()
        headers = ['Token', 'Score']
        table_data = []
        for token, score in zip(sentence_tokenized, attentions):
            table_data.append([token, "{:.3f}".format(score)])
        return tabulate.tabulate(table_data, headers)

    def return_selection_wn_fallback(self, input_sentence_doc, initial_threshold=2):
        # this is a special function that increases the attention threshold if none of the 
        # selected tokens is supported by wordnet

        pos_supported_by_wn = ['NOUN', 'ADJ','ADV','VERB']

        if not any([token.pos_ in pos_supported_by_wn for token in input_sentence_doc]):
            # if none of the tokens are supported by wn, return empy list
            logger.info('None of the tokens in the input sentence are supported by wordnet')
            return []

        wn = WNSubstitutor.from_spacy_doc(input_sentence_doc)
        
        # get all attention scores
        selected_tokens_score_index = self.return_selected_tokens()

        slices = []
        for _, _, i in selected_tokens_score_index:
            slices.append(slice(i, i+1))

        supported_slices = []

        for token_slice in slices:
            # create selection for slice
            selection = Selection(token_slice)
            # assert whether it is supported by wordnet
            if wn.selection_supports_wn(selection):
                supported_slices.append(token_slice)
            if len(supported_slices) == initial_threshold:
                break

        if len(supported_slices) > 0:
            selections = []
            for set in self.get_powerset_from_slices(supported_slices, 2):
                selections.append(Selection(set))
            tokens = [selection.return_slices_as_sequences(input_sentence_doc) for selection in selections]
            logger.info('The tokens {} are supported by wordnet'.format(tokens))
            return selections
        else:
            logger.info('None of the tokens in the input sentence are supported by wordnet')
            return []


    def return_selected_tokens(self):
        attention = self.get_attention_score().squeeze().cpu().numpy()

        # TODO: if input is only a single token, attention is a 1-D ndarray
        # wrap it in a list to make it iterable for the loop below
        # this is a hack, find where attention tensors are squeezed upstream
        if len(attention.shape) == 0:
            attention = attention.reshape(1)

        selected_tokens_score_index = []

        for i, (token, score) in enumerate(zip(self.input_tokenized, attention)):
            selected_tokens_score_index.append((token, score, i))

        selected_tokens_score_index.sort(key=lambda t: t[1], reverse=True)

        logger.info('Tokens {} selected via attention in decreasing order.'.format([t[0] for t in selected_tokens_score_index]))
        return selected_tokens_score_index


    def return_selection(self, threshold=2):
        selected_tokens_score_index = self.return_selected_tokens()

        if len(selected_tokens_score_index) > threshold:
            selected_tokens_score_index = selected_tokens_score_index[:threshold]

        slices = []
        for _, _, i in selected_tokens_score_index:
            slices.append(slice(i, i+1))

        selections = []
        for set in self.get_powerset_from_slices(slices, 2):
            selections.append(Selection(set))

        logger.info('{} selections created first {} tokens'.format(len(selections), threshold))

        return selections

    def is_emotional(self):

        selections = self.return_selection()
        if len(selections) == 0:
            return False
        
        return True



if __name__ == '__main__':
    model_name = 'tec'
    pretrained_dir = global_config['directories']['pretrained_dir']

    model_path = os.path.join(pretrained_dir, model_name + "_emocl.pt")
    fields_path = os.path.join(pretrained_dir, model_name + "_fields.dill")

    model = torch.load(model_path, map_location='cpu')

    with open(fields_path, 'rb') as f:
        fields = dill.load(f)

    # create emocl instance
    emocl = EmotionClassifier(model, fields['TEXT'], fields['LABEL'])

    single_word = ['Test']

    sel = AttentionSelector(single_word, emocl)

    print(sel.return_selection())
