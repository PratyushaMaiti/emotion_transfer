# Emotion Style Transfer in Text

## Prerequisites

### Get the repository
Clone the current repository using 

    > git clone git@clarin06.ims.uni-stuttgart.de:DavidHelbig/emotion-transfer.git

### Environment

Make sure you have at least **Python 3.7** installed. The creation of a virtual environment is heavily recommended when working with the repository:

Move to the desired directory where you want to create your environment (for example the root folder of the repository) and run 

    > python3 -m venv env

This will create a virtual environment in the subfolder *env/*. 
Activate the virtual environment with 

    > source env/bin/activate

Now, with the virtual environment activated in the current shell session, move to the repository root (if you're not already there) and install the necessary packages with 

    > pip3 install -r requirements.txt

**All ensuing commands must be run with the virtual environment activated** 
(Usually, the session is closed when the terminal window is closed. The virtual environment can be close manually with the `deactivate` command)

### Datasets

Run the [The Unified Emotion Dataset](https://github.com/sarnthil/unify-emotion-datasets/tree/master/datasets) script. Put the resulting `unified-dataset.jsonl` file into the *datasets/* subfolder of the cloned repository.

We further use NLTK to interface wordnet.
For the case you don't already have downloaded it for a different project, the wordnet data can be downloaded for NLTK with the following command:

```
python -m nltk.downloader wordnet
```
(check https://www.nltk.org/data.html for more info on downloading data for NLTK).

### Additional dependencies

emotion_transfer makes use of Spacy for different tasks. It requires the *en_core_web_sm* module. It can be downloaded running

    python3 -m spacy download en_core_web_sm

We further use WordNET via NLTK. 

### Train at least one emotion classifier module

Part of the minimum requirements is to train at least one emotion classifier module. 
The definition of the metaparameters takes place in config files, which are stored in the *configs/emocl_train/* subdirectory. 
Refer to the comments in the *tec.cfg* file to get an overview of the available settings.
Use the config as a blueprint and set your desired settings (you migth want to change the *name* and *dataset* parameters).
You can use any filename you want for the config.
Once you fixed your settings, run the training script from the repository root directory, which takes the path to the config file as its only parameter.

    > python3 -m objective.emocl.train configs/emocl_train/name_of_your_config.cfg

The config used to train the classifier in the paper is located at `configs/emocl_train/tec.cfg`. It further requires pre-trained 300-dimensional Twitter embeddings provided at https://github.com/cbaziotis/ntua-slp-semeval2018. Downlaod the 300-dimensional embeddings from there and put them in `objective/emocl/nn/embeddings` if you would like to train on this config.

The emotion classifier will now start training with the given parameters. Once training is finished, the trained module will be placed in the *pretrained/* subdirectory of the repository root.
The training script saves two binary files: One *name_emocl.pt* file which contains the module itself including the weights, and one *name_fields.dill* file which contains mapping information for input tokens to embeddings and numericalized labels.

## Define your pipeline

The definition of modules in a pipeline also happens via config files. Please refer to the commented *configs/pipelines/ppl_tec_wna_wn.cfg* file to get a blueprint and and an overview of the supported modules.

Currently available: (corresponding cfg value in brackets)

| Selection | Substitution | Objective |
| --------- | ------------ | --------- |
| Brute forece single (bf_single) | Wordnet (wordnet) | Emotion score (emotion) |
| Brute force all (bf_all) | Paraphrase Database (ppdb) | Sentence similarity (sim) |
| Brute force single (bf_single) | | |
| Brute force threshold (bf_threshold) | | |
| Attention (attention) | | |

Give the config file a speaking name (it will be referenced in the module output) and save it in the *configs/pipelines/* subdirectory.

## Run the pipeline using the main script 

Once you defindet your pipeline in a config, you can run the pipeline using the main script.

The script itself has a *--help* flag which displays the available parameters:

    > python3 emotion_transfer.py --help

Please refer to the help text (copied below).
```
Usage: emotion_transfer.py [OPTIONS] TARGET_EMOTION

  Emotion transfer command line interface.

  Input is read from stdin. Multiple sentences are supported and are split
  by newline.

      Example:

          cat mysentences.txt | python emotion_transfer.py [OPTIONS]
          TARGET_EMOTION

  Alternatively, a text file with sentences can be passed via the --input
  option:

      Example:

          python emotion_transfer.py --input mysentences.txt [OPTIONS]
          TARGET_EMOTION

  It is recommended to use a configuration file specifying the components to
  be used, which should be passed with the --config option parameter. If not
  config file is provided, component definitions must be passed as option
  parameters (see below).

Options:
  --input FILENAME        Provide a text file with input sentences. If not
                          provided, trying to read from stdin.
  --config PATH           A config file defining the components to be used. If
                          provided, it will override all component definitions
                          that are passed as command line parameters.
                          [required]
  -k INTEGER
  --output FILENAME       The path to the file where results should be written
                          to.
  --format [console|csv]  The output format.
  --verbose
  --help                  Show this message and exit.
```