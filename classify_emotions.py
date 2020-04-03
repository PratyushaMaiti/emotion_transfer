from objective.emotion import EmotionClassifier
from cfgparser import global_config

import click
import csv
import dill
import os
import torch

@click.command()
@click.argument('pretrained_model')
@click.argument('input_file', type=click.File('r'))
@click.argument('output_file', type=click.File('w'))
def main(pretrained_model, input_file, output_file):
    pretrained_dir = global_config['directories']['pretrained_dir']
    model_path = os.path.join(pretrained_dir, pretrained_model + "_emocl.pt")
    fields_path = os.path.join(pretrained_dir, pretrained_model + "_fields.dill")
    model = torch.load(model_path)
    model.eval()

    with open(fields_path, 'rb') as f:
        fields = dill.load(f)

    emocl = EmotionClassifier(model, fields['TEXT'], fields['LABEL'])

    input_sentences = [sentence.strip() for sentence in input_file.readlines()]

    scores = emocl.get_scores(input_sentences)

    fieldnames = ['sentence'] + list(fields['LABEL'].vocab.stoi.keys())

    writer = csv.DictWriter(output_file, fieldnames)

    writer.writeheader()

    for sentence, score_dict in zip(input_sentences, scores):
        row_dict = score_dict.copy()
        row_dict['sentence'] = sentence
        writer.writerow(row_dict)

if __name__ == '__main__':
    main()