import json
import logging
import os
import random
import tqdm

from cfgparser import config_dir, global_config, parse_config
from datetime import datetime
from pipeline import Pipeline

logger = logging.getLogger('human_eval')
logger.setLevel(logging.INFO)

warning_logger = logging.getLogger()
warning_logger.setLevel(logging.WARNING)

log_path = os.path.join(global_config['directories']['log_dir'], "human_eval.log")
warning_log_path = os.path.join(global_config['directories']['log_dir'], "human_eval_warnings.log")

fh = logging.FileHandler(filename=log_path, mode='a')
fh_warnings = logging.FileHandler(filename=warning_log_path, mode='a')
fmt = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(fmt)
fh_warnings.setFormatter(fmt)
logger.addHandler(fh)
warning_logger.addHandler(fh_warnings)


class HumanExperimenter:

    def __init__(self, configs, random_seed):

        self.configs = configs
        self.outputs = []
        self.duplicates = []
        self.id_config_mapping = {}
        self.num_duplicates = 0
        random.seed(random_seed)
        logger.info("Random seed initialized with {}".format(random_seed))

    @classmethod
    def read_state_from_file(cls, json_file, random_seed):
        with open(json_file, 'r') as f:
            content = json.load(f)

        id_config_mapping = content['id_config_mapping']
        configs = list(set(id_config_mapping.values()))

        exp = cls(configs, random_seed)
        
        exp.id_config_mapping = id_config_mapping
        exp.duplicates = content['duplicates']
        exp.outputs = content['outputs']

        for obj in content['outputs']:
            if any(obj['outputs'][c_name].get('is_duplicate') for c_name in obj['outputs'].keys()):
                exp.num_duplicates += 1

        logger.info('Experiment state loaded from {}'.format(json_file))
        return exp


    def id_generator(self):
        ids = random.sample(range(10000), 1000)
        if self.id_config_mapping != {}:
            for id in self.id_config_mapping.values():
                if id in ids:
                    ids.remove(id)
        n = 0
        while n < len(ids):
            yield str(ids[n]).zfill(4)
            n += 1

    def run_experiments(self, sentences):

        if self.outputs != []:
            logger.info('Append to existing output sentences')

        gen = self.id_generator()
        config_paths = [os.path.join(config_dir, 'pipelines', config) for config in self.configs]
        parsed_configs = [parse_config(config_path) for config_path in config_paths]

        for s in sentences:
            is_duplicate_sentence = False
            for o in self.outputs:
                if s == o['input_sentence']:
                    is_duplicate_sentence = True

            if not is_duplicate_sentence:
                output_obj = {}
                output_obj['input_sentence'] = s.strip()
                target_emotion = random.choice(['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust'])
                output_obj['target_emotion'] = target_emotion

                output_obj['outputs'] = {}
                self.outputs.append(output_obj)

        for c_name, c_parsed in zip(self.configs, parsed_configs):
            pipe = Pipeline.from_config(c_parsed, output_method=None, device='cpu')
            logger.info('Created new pipeline for config {} with parameters\n{}'.format(c_name, pipe.return_parameters_table()))
            for o in tqdm.tqdm(self.outputs):
                if c_name not in o['outputs'].keys():
                    try:
                        variations, scores = pipe.process_pipeline(o['input_sentence'], o['target_emotion'])
                    except TypeError:
                        logging.warning('No output produced for sentence {} using config {}'.format(o['input_sentence'], c_name))
                        o['outputs'][c_name] = None
                        continue
                    else:
                        output_id = next(gen)
                        self.id_config_mapping[output_id] = c_name
                        # pick the highest ranking variation
                        is_duplicate = False
                        for config, output_dict in o['outputs'].items():
                            if output_dict['output_sentence'] == variations[0]:
                                logger.warning('Two identical outputs created: {}'.format(variations[0]))
                                self.duplicates.append((variations[0], output_id, output_dict['output_id']))
                                self.num_duplicates += 1
                                is_duplicate = True

                        o['outputs'][c_name] = {'output_sentence': variations[0], 'output_id': output_id, 'is_duplicate': is_duplicate}


    def write_to_file(self, infix='extendend'):
        json_obj = {'id_config_mapping': self.id_config_mapping, 'outputs': self.outputs, 'duplicates': self.duplicates}
        datestring = datetime.now().strftime('%Y-%m-%d--%H-%M-%S')

        output =  datestring + '_' + infix + '_human_eval_sentences.json'

        with open(output, 'w') as f:
            json.dump(json_obj, f, indent=2)

        logger.info('Outputs written to {}'.format(output))


if __name__ == '__main__':

    state_file = '2019-10-23--22-15-18_human_eval_sentences_complete.json'

    sentence_file = '2019-10-14_tec_extension.txt'


    with open(sentence_file, 'r') as f:
        sentences = f.readlines()

    seed = 42
    exp = HumanExperimenter.read_state_from_file(state_file, seed)
    exp.run_experiments(sentences)
    exp.write_to_file()
