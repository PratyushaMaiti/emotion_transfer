import logging
import os
import time
from collections.abc import Sequence

import click
import dill
import torch

from cfgparser import parse_config
from objective import emotion
from inout.output import OutputGenerator
from pipeline import make_pipeline
from inout.output import OutputGenerator



def get_input(ctx, param, file):
    if not file and not click.get_text_stream('stdin').isatty():
        return click.get_text_stream('stdin').read().splitlines()
    else:
        return file.read().splitlines()


@click.command()
@click.option('--input', type=click.File(), callback=get_input, help="Provide a text file with input sentences. If not provided, " \
    "trying to read from stdin.")
@click.argument('target_emotion')
@click.option('--config', required=True, type=click.Path(exists=True), multiple=True, help="A config file defining the components to be used. If provided, " \
    "it will override all component definitions that are passed as command line parameters.")
@click.option('-k', type=int, default=5)
@click.option('--output', type=click.File('w'), help='The path to the file where results should be written to.')
@click.option('--format', type=click.Choice(['console', 'csv']), default='console', help='The output format.')
@click.option('--verbose', is_flag=True)
def cli(input, target_emotion, config, k, verbose, output, format):
    """ Emotion transfer command line interface. The single mandatory argument is the target emotion. 

    Input is read from stdin. Multiple sentences are supported and are split by newline.

        Example:

            cat mysentences.txt | python emotion_transfer.py [OPTIONS] TARGET_EMOTION
    
    Alternatively, a text file with sentences can be passed via the --input option:

        Example:

            python emotion_transfer.py --input mysentences.txt [OPTIONS] TARGET_EMOTION
    
    It is recommended to use a configuration file specifying
    the components to be used, which should be passed with the --config option parameter. If not config file is provided, component definitions
    must be passed as option parameters (see below)."""


    # initialize logger
    log_level = logging.ERROR
    if verbose:
        log_level = logging.INFO
    
    logging.basicConfig(format='%(asctime)s:%(levelname)s - %(message)s', level=log_level)

    if input is None:
        raise click.UsageError('Input muste be either provided via --input option as argument or by piping into stdin.')

    if not isinstance(input, list):
        input = [input]

    if not isinstance(config, Sequence):
        config = [config]

    output_rows = []
    fieldnames = ['Original Sentence',  'Sentence Variation', 'Target Scores'] 
    target_scores = None
    for c in config:
        config = parse_config(c)
        model = config['pretrained']['emocl_model']
        section = config['pipeline']
        selection = section['selection_method']
        substitution = section['substitution_method']
        objectives = section['objective'].split(",")
        if format is None:
            format = section['output_format']
        if output is None and format != 'console':
            output = section['output_filename']

        if target_scores is None:
            target_scores = objectives.copy()
        # right now, scores have to match in all configs for output reasons.
        # This could be changed later, for example by creating different output files per config
        elif target_scores is not None and target_scores != objectives:
            logging.warning('Scores definded in config {} are different from previous config. Skipping.'.format(c))
            continue

        weights = None if not 'weights' in config else config['weights']

        pipeline = make_pipeline(model, selection, substitution, objectives, weights, config, format, output)
        duration_cumulative = 0
        for i, sentence in enumerate(input, start=1):
            logging.info('Processing sentence number {}'.format(i))
            time_start = time.time()
            try:
                variations, scores = pipeline.process_pipeline(sentence, target_emotion)
            except TypeError as err:
                logging.exception(err)
                logging.warning('No output produced for sentence "{}" and config {}'.format(sentence, c))
            else:
                _, config_name = os.path.split(c)
                fieldnames, rows = OutputGenerator.to_tabular_format(sentence, i, variations, scores, target_scores, target_emotion, config_name, k=k)
                output_rows += rows
                time_end = time.time()
                logging.info('Sentence {} processed in {:.2f} seconds.'.format(i, time_end - time_start))
                duration_cumulative += time_end - time_start
                logging.info('Average duration: {:.2f} seconds.'.format(duration_cumulative/i))

    if format == 'console':
        OutputGenerator.to_screen(fieldnames, output_rows)
    elif format == 'csv':
        OutputGenerator.to_csv(fieldnames, output_rows, output)

if __name__ == '__main__':
    cli()