import argparse
import logging
import os

import click
import dill
import torch
import tabulate

from cfgparser import global_config
from selection.bruteforce import BruteForceSelector
from selection.attention import AttentionSelector
from selection.dictionaries import NRC_Selector, WNA_Selector
from selection.utils.nrc_interface import NRC_Interface
from substitution.wordnet import WNSubstitutor, POSNotSupported
from substitution.ppdb import PpdbSubstitutor
from embedding_space.embedding_space_extractor import EmbeddingSpace
from substitution.embedding_space import EmbeddingSpaceSubstitutor, InformedEmbeddingSpaceSubstitutor, OutOfVocabulary
from objective import emotion
from objective.sim.similarity_bert import SimilarityScorerBert
from objective.lm.get_perplexity import PerplexityScorer
from preprocessing import SpacyPreprocessor
from inout import output


logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Pipeline:

    def __init__(
        self, 
        emocl_model, 
        selection_component,
        substitution_component, 
        objective_components,
        weights,
        config,
        output_method='console',
        output_filename=None):


        if isinstance(emocl_model, emotion.EmotionClassifier):
            self.emocl_model = emocl_model
        else:
            raise TypeError('emocl model must be a EmotionClassifier instance.')

        self.selection_component = selection_component
        self.substitution_component = substitution_component
        self.objective_components = objective_components

        if not isinstance(self.objective_components, list):
            self.objective_components = [self.objective_components]

        self.weights = weights
        if self.weights is not None:
            if sum(map(float, self.weights.values()))  != 1:
                logger.warning('The sum of all weights does not add up to 1')
            if not all(key in self.objective_components for key in self.weights.keys()):
                logger.warning('The keys for weights in the config file do not correspond to objective components!')
        else:
            self.weights = {func_name: 1/len(self.objective_components) for func_name in self.objective_components}
            logger.info('No weights defined, using equal weights for all scoring functions.')

        self.output_method = output_method
        if output_method is None:
            self.output_method = 'console'
        self.output_filename = output_filename
        self.config = config

        # instantiate model classes if necessary
        if 'sim' in self.objective_components:
            self.sims = SimilarityScorerBert()

        if 'lm'  in self.objective_components:
            self.ps = PerplexityScorer()

        self.initialize_parameters()


        if self.substitution_component == 'distr_informed' or self.substitution_component == 'distr_uninformed':
            self.initialize_embedding_spaces(self.substitution_component == 'distr_informed')


    def initialize_parameters(self):
        # read additional parameters if necessary
        if self.selection_component == 'attention':
            self.selection_threshold = int(self.config['pipeline'].get('selection_threshold', 2))

        if self.substitution_component == 'wordnet':
            if 'wordnet' in self.config:
                self.move_up = int(self.config['wordnet'].get('move_up', 0))
                self.move_down = int(self.config['wordnet'].get('move_down', 0))
        elif self.substitution_component == 'distr_informed':
            self.n_neighbors = int(self.config['distr_informed'].get('n_neighbors', 100))
            self.k_closest = int(self.config['distr_informed'].get('k_closest', 20))
        elif self.substitution_component == 'distr_uninformed':
            self.n_neighbors = int(self.config['distr_uninformed'].get('n_neighbors', 100))


    def initialize_embedding_spaces(self, informed):
        
        # if no paths to a model or indexer are given in the config,
        # try to build the embedding space from the embedding weights in the emocl model
        try:
            model_name = self.config['pretrained']['model_name']
            indexer_name = self.config['pretrained']['indexer_name']
        except KeyError:
            model_name = None
            indexer_name = None

        # try to read the number of trees parameter from config
        
        if model_name is not None and indexer_name is not None:
            model_path = os.path.join(global_config['directories']['pretrained_dir'], self.config['distr']['model_name'])
            indexer_path = os.path.join(global_config['directories']['pretrained_dir'], self.config['distr']['indexer_name'])

            self.embedding_space = EmbeddingSpace.load_from_files(model_path=model_path, indexer_path=indexer_path)
            logger.info('Embedding space loaded from files {} and {}'.format(model_name, indexer_name))
        else:
            logger.info('No embedding space related files found, build embedding space from emotion classifier embeddings')
            # try to read the n_trees setting from either a dist_informed or dist_uninformed config section
            if informed:
                n_trees = int(self.config['distr_informed'].get('n_trees', 10))
            else:
                n_trees = int(self.config['distr_uninformed'].get('n_trees', 10))
            entities = self.emocl_model.text_field.vocab.itos
            embedding_weights = EmbeddingSpace.return_embedding_weights(self.emocl_model.model)
            self.embedding_space = EmbeddingSpace.build_from_keyed_vectors(
                entities=entities, 
                embedding_weights=embedding_weights,
                annoy_trees=n_trees)

        if informed:
            logger.info('Loading emotion lexicon for emotion informed substitution')

            # initialize lexicon
            # TODO: change from hardcoded to config option
            self.nrc = NRC_Interface(path='datasets/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
            self.nrc.parse()


    @classmethod
    def from_config(clf, parsed_config, output_method, output_filename=None, device=DEVICE):

        pretrained_dir = global_config['directories']['pretrained_dir']
        model_name = parsed_config['pretrained']['emocl_model']

        model_path = os.path.join(pretrained_dir, model_name + "_emocl.pt")
        fields_path = os.path.join(pretrained_dir, model_name + "_fields.dill")

        # load the emotion classifier and fields
        model = torch.load(model_path).to(device)
        model.eval()

        with open(fields_path, 'rb') as f:
            fields = dill.load(f)

        emocl = emotion.EmotionClassifier(model, fields['TEXT'], fields['LABEL'])

        pipeline_section = parsed_config['pipeline']

        weights = None if not 'weights' in parsed_config else parsed_config['weights']

        pipeline = clf(
            emocl_model=emocl,
            selection_component=pipeline_section['selection_method'],
            substitution_component=pipeline_section['substitution_method'],
            objective_components=pipeline_section['objective'].split(','),
            weights=weights,
            config=parsed_config,
            output_method=output_method,
            output_filename=output_filename
        )

        return pipeline

        
        
    def initialize_selection(self, input_sentence_doc, input_sentence_tokenized):

        if self.selection_component == 'attention' and self.substitution_component == 'wordnet':
            selector = AttentionSelector(input_sentence_tokenized, self.emocl_model)
            logger.info('Using attention selection with wordnet-fallback')
            self.select_method = lambda: selector.return_selection_wn_fallback(input_sentence_doc, initial_threshold=self.selection_threshold)
        elif self.selection_component == 'bf_single':
            selector = BruteForceSelector.from_spacy_doc(input_sentence_doc)
            self.select_method = selector.select_one_token
        elif self.selection_component == 'bf_all':
            selector = BruteForceSelector.from_spacy_doc(input_sentence_doc)
            self.select_method = selector.select_all_combinations
        elif self.selection_component == 'bf_threshold':
            selector = BruteForceSelector.from_spacy_doc(input_sentence_doc)
            self.select_method = selector.select_all_combinations_with_threshold
        elif self.selection_component == 'attention':
            selector = AttentionSelector(input_sentence_tokenized, self.emocl_model)
            self.select_method = lambda: selector.return_selection(threshold=self.selection_threshold)
        elif self.selection_component == 'wna':
            selector = WNA_Selector.from_spacy_doc(input_sentence_doc)
            self.select_method = selector.return_selections
        elif self.selection_component == 'nrc':
            selector = NRC_Selector.from_spacy_doc(input_sentence_doc)
            self.select_method = selector.return_selections
        else:
            raise ValueError('"{}" is not an available selection method'.format(self.selection_component))

    def initialize_substitution(self, input_sentence_doc, target_emotion):

        # check the 'conjugate' setting in the config
        conjugate_substitutes = self.config['pipeline'].getboolean('conjugate', False)

        logger.info('Automated conjugation of substitutes set to {}'.format(conjugate_substitutes))

        if self.substitution_component == 'wordnet':
            if 'wordnet' in self.config:
                self.move_up = int(self.config['wordnet'].get('move_up', 0))
                self.move_down = int(self.config['wordnet'].get('move_down', 0))

            substitutor = WNSubstitutor.from_spacy_doc(input_sentence_doc)
            self.substitute_method = lambda selection: substitutor.create_variations(selection, self.move_up, self.move_down, conjugate_substitutes)
        elif self.substitution_component == 'ppdb':
            substitutor = PpdbSubstitutor.from_spacy_doc(input_sentence_doc)
            self.substitute_method = lambda selection: substitutor.create_variations(selection, conjugate_substitutes)
        elif self.substitution_component == 'distr_informed':

            self.n_neighbors = int(self.config['distr_informed'].get('n_neighbors', 100))
            self.k_closest = int(self.config['distr_informed'].get('k_closest', 20))

            wordlists = self.nrc.wordlists
            embedding_substitutor = InformedEmbeddingSpaceSubstitutor.from_spacy_doc(input_sentence_doc, self.embedding_space, wordlists)
            self.substitute_method = lambda selection: embedding_substitutor.create_variations(
                selection, 
                emotion=target_emotion,
                n_neighbors=self.n_neighbors,
                n_least_distance_to_centroid=self.k_closest,
                conjugate_substitutes=conjugate_substitutes)
        elif self.substitution_component == 'distr_uninformed':

            self.n_neighbors = int(self.config['distr_uninformed'].get('n_neighbors', 100))
            embedding_substitutor = EmbeddingSpaceSubstitutor.from_spacy_doc(input_sentence_doc, self.embedding_space)

            self.substitute_method = lambda selection: embedding_substitutor.create_variations(selection, self.n_neighbors, conjugate_substitutes=False)

        else:
            raise ValueError("'{}' is not a supported substitution component.".format(self.substitution_component))

    def initialize_objective(self, input_sentence, target_emotion):
        self.scoring_functions = []

        if 'emotion' in self.objective_components:
            self.scoring_functions.append((lambda variations: self.emocl_model.get_score_for_target_emotion(variations, target_emotion), 'emotion'))
        else:
            logger.warn('Not using emotion score in objective function^.')

        if 'sim' in self.objective_components:
            self.scoring_functions.append((lambda variations: self.sims.return_similarity_scores(input_sentence, variations), 'sim'))

        if 'lm'  in self.objective_components:
            self.scoring_functions.append((self.ps.return_lm_scores, 'lm'))


    def initialize_components(self, input_sentence, target_emotion):
        """ Initialize components with a given input_sentence """

        prepr = SpacyPreprocessor()
        # Preprocess input
        input_sentence_doc = prepr.process_input(input_sentence)
        input_sentence_tokenized = [token.lower_ for token in input_sentence_doc]

        self.initialize_selection(input_sentence_doc, input_sentence_tokenized)
        self.initialize_substitution(input_sentence_doc, target_emotion)
        self.initialize_objective(input_sentence, target_emotion)

    def perform_substitution(self, selections):
        variations = []
        for s in selections:
            try:
                variations += self.substitute_method(s)
            except POSNotSupported as err:
                pass
#                logger.warning('POS "{}" is not supported by WordNet'.format(err.args[0]))
            except OutOfVocabulary as err:
                pass
#                logger.warning(err)

        return variations

    def perform_selection(self):
        selection = self.select_method()
        # if selection is not a list (as returned for example by the brute force methods),
        # wrap it in one
        if not isinstance(selection, list):
            selection = [selection]

        return selection

    def return_scores(self, variation_sentences, variation_sentences_untokenized):
        # initialize empty dictionary
        merged_scores = [{} for _ in range(len(variation_sentences))]
        # each scoring function returns a dictionary
        # merge dictionaries of all scoring functions into one:
        for function, function_name in self.scoring_functions:
            if function_name == 'emotion':
                variations = variation_sentences
            elif function_name == 'sim' or function_name == 'lm':
                variations = variation_sentences_untokenized
            logger.info("Calculate scores for variations using {}.".format(function_name))
            score_dicts = function(variations)
            for i, s_d in enumerate(score_dicts):
                merged_scores[i].update(s_d)

        return merged_scores

    def sort_by_target_emotion(self, sentences, scores):
        zipped = zip(sentences, scores)
        zip_sorted = sorted(zipped, key=lambda t: t[1]['emotion'], reverse=True)
        # return unzipped list
        return list(zip(*zip_sorted))

    def sort_by_maximized_scores(self, sentences, scores, to_maximize):
        logging.info("Sorting: maximizing scores {}".format(to_maximize))
        zipped = zip(sentences, scores)
        zip_sorted = sorted(zipped, key=lambda t: sum([t[1][score] for score in to_maximize]))
        return list(zip(*zip_sorted))

    def sort_by_weighted(self, sentences, scores):
        logging.info("Sort by weighted scores")
        zipped = zip(sentences, scores)
        zip_sorted = sorted(zipped, key=lambda t: sum([t[1][objective] * float(self.weights[objective]) for objective in [sf[1] for sf in self.scoring_functions]]), reverse=True)
        return list(zip(*zip_sorted))

    def process_pipeline(self, input_sentence, target_emotion, sort_method=None):
        if sort_method is None:
            sort_method = self.sort_by_weighted
        logger.info('Start initalization of components.')
        self.initialize_components(input_sentence, target_emotion)
        logger.info('Start selection procedure.')
        selections = self.perform_selection()
        logger.info('Perform substitution.')
        variations = self.perform_substitution(selections)
        logger.info('Finished substitution.')
        logger.info('{} variation sentences in total'.format(len(variations)))

        # un-tokenize variations (required for sentence similarity right now)
        variations_untokenized = [" ".join(s) for s in variations]

        try:
            scores = self.return_scores(variations, variations_untokenized)
        except (ValueError,TypeError) as err:
            logging.error(err)
            logging.warning('Could not get scores for variations.')
            return None

        # TODO: devise a method that ranks according to multiple (weighted) scores
        variations_untokenized, scores = sort_method(variations_untokenized, scores)
        return variations_untokenized, scores

    def process_output(self, original_sentence, variation_sentences, scores):
        target_scores = self.objective_components.copy()
        if 'emotion' in target_scores:
            target_scores.remove('emotion')
            target_scores.append(self.target_emotion)
        generator = output.OutputGenerator(original_sentence, variation_sentences, scores, target_scores=target_scores, k=5)
        if self.output_method == 'console':
            generator.to_screen()
        elif self.output_method == 'csv':
            generator.to_csv(self.output_filename)
        else:
            logger.error('{} is not a valid output method'.format(self.output_method))

    def return_parameters_table(self):

        parameters = [
                ['Pretrained Model', self.config['pretrained']['emocl_model']],
                ['Selection Component', self.selection_component],
                ['Substitution Component', self.substitution_component],
                ['Objective Components', self.objective_components],
                ['Weights', str({component: '{:.2f}'.format(float(weight)) for component, weight in self.weights.items()})]
            ]

        if self.selection_component == 'attention':
            parameters.append(['Selection threshold', self.selection_threshold])

        if self.substitution_component == 'wordnet':
            parameters.append(['Wordnet range', str({'move_up': self.move_up, 'move_down': self.move_down})])
        elif self.substitution_component == 'distr_uninformed':
            parameters.append(['Number of neighbors', self.n_neighbors])
        elif self.substitution_component == 'distr_informed':
            parameters.append(['Number of neighbors', self.n_neighbors])
            parameters.append(['Number of closest to centroid', self.k_closest])

        return tabulate.tabulate(
            tabular_data=parameters,
            headers=['Parameter', 'Value'])


def get_fields(emocl_model_name):
    fields_filename = emocl_model_name + '_fields.dill'
    try:
        with open(os.path.join(global_config['directories']['pretrained_dir'], fields_filename), 'rb') as f:
            fields = dill.load(f)
    except FileNotFoundError as err:
        logger.exception(err)
        raise
    return fields


def get_emocl_model(emocl_model_name):
    model_filename = emocl_model_name + '_emocl.pt'
    try:
        model = torch.load(os.path.join(global_config['directories']['pretrained_dir'], model_filename))
        model = model.eval()
    except FileNotFoundError as err:
        logger.exception(err)
        raise
    fields = get_fields(emocl_model_name)
    ec = emotion.EmotionClassifier(model, fields['TEXT'], fields['LABEL'])
    return ec

def make_pipeline(emocl_model_name, selection_component, substitution_component, objective_components, weights, config, output_method, output_filename=None):

    emocl_model = get_emocl_model(emocl_model_name)

    pipeline = Pipeline(
        emocl_model, 
        selection_component=selection_component, 
        substitution_component=substitution_component, 
        objective_components=objective_components,
        output_method=output_method,
        weights=weights,
        config=config,
        output_filename=output_filename,
        )
    return pipeline
