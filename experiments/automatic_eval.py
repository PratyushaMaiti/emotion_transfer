import os
import json
import logging
import random
import time
from datetime import datetime

from tqdm import tqdm

from cfgparser import global_config, config_dir, parse_config
from pipeline import Pipeline
from objective.emocl.ued_handler import UEDLoader

loader = UEDLoader(path=os.path.join(global_config['directories']['datasets_dir'], "unified-dataset.jsonl"))

logger = logging.getLogger('auto_eval')
logger.setLevel(logging.INFO)

warning_logger = logging.getLogger()
warning_logger.setLevel(logging.WARNING)

log_path = os.path.join(global_config['directories']['log_dir'], "auto_eval.log")
warning_log_path = os.path.join(global_config['directories']['log_dir'], "auto_eval_warnings.log")

fh = logging.FileHandler(filename=log_path, mode='a')
fh_warnings = logging.FileHandler(filename=warning_log_path, mode='a')
fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(fmt)
fh_warnings.setFormatter(fmt)
logger.addHandler(fh)
warning_logger.addHandler(fh_warnings)

# filter for the twitter emotion corpus
tec = loader.filter_datasets(source="tec")

emotions = tec.return_emotion_classes()

# set the random seed to get reproducible results
r_seed = 1502
random.seed(r_seed)

# sample 1000 sentences
sample = random.sample([example.text for example in tec.examples], 1000)

configs = ['baseline.cfg', 'attn_wn.cfg', 'attn_distr.cfg']
results_dir = global_config['directories']['results_dir']

datestring = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')

sample_sentence_path = os.path.join(results_dir, datestring + "_sample_sentences.txt")

# write sentences to output file
with open(sample_sentence_path, 'w') as f:
    for sentence in sample:
        f.write(sentence + "\n")

logger.info('Sample sentences written to {}'.format(sample_sentence_path))


results = {config: {} for config in configs}
results['sample_size'] = len(sample)

logger.info('Start logging for random sample of {} sentences from tec (seed={})'.format(len(sample), r_seed))
start_time = time.time()

for config in configs:
    config_path = os.path.join(config_dir, 'pipelines', config)
    config_basename = os.path.splitext(config)[0]
    
    config_parsed = parse_config(config_path)

    config_parsed['pipeline']['objective'] = 'emotion'

    pipeline = Pipeline.from_config(config_parsed, 'csv')
    logger.info('Created new pipeline for config {}:\n{}'.format(config, pipeline.return_parameters_table()))


    for i, sentence in enumerate(tqdm(sample), 1):
        for emotion in emotions:

            # return scores sorted by target emotion score
            scores = pipeline.process_pipeline(sentence, emotion, sort_method=pipeline.sort_by_target_emotion)

            if scores is not None:

                highest_score = scores[1][0]['emotion']

                results[config].setdefault(emotion, 0)

                # aggregate highest emotion score
                results[config][emotion] += highest_score

            else:
                logger.warning('No output produced for sentence {}, config {} and emotion {}'.format(i, config, emotion))

end_time = time.time()


for config_name in configs:
    for emotion in emotions:
        results[config_name][emotion] /= len(sample)

logger.info('Evaluation finished in {:.2f} seconds'.format(end_time-start_time))

output_file = os.path.join(results_dir, datestring + "_automatic_eval.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

logger.info('Reults saved to {}'.format(output_file))