# Essentia introduction
Essentia is a pipeline that mines domain-specific paraphrases in an unsupervised, graph-based manner.
It takes as input a set of sentences of the same topic, such as "asking directions" or "booking a restaurant",
and output a set of candidate paraphrases of which the majority are domain-specific paraphrases (e.g., {"Book a restaurant", "Reserve a restaurant"}).

For more details, please check the following publication:\
[Essentia: Mining Domain-Specific Paraphrases with Word-Alignment Graphs](https://arxiv.org/abs/1910.00637)

# Environment setup:
* Use Python 3.7.

* Install packages specified in `requirements.txt`.

* Download spacy pre-trained English models:\
`python -m spacy download en`\
`python -m spacy download en_core_web_lg`

# Use Essentia to mine domain-specific paraphrases:

## Set up the word aligner:
Our word aligner is a modified version of the word aligner in the following repository:\
https://github.com/rameshjesswani/Semantic-Textual-Similarity

* Clone the word aligner repository to local.
* Copy the *monolingualWordAligner* directory in the repository into `/PATH/TO/ESSENTIA/scripts/essentia/`.
* We reimplement the word aligner to make it run with Python 3. Copy all files under `/PATH/TO/ESSENTIA/scripts/essentia/word_aligner` into `/PATH/TO/ESSENTIA/scripts/essentia/monolingualWordAligner`.


## Run Essentia:
* Run `python scripts/essentia/mine_para.py --dir /DIR/OF/INPUT/FILES --output /DIR/OF/OUTPUT/PARAPHRASES` at the root directory.

By default, `mine_para.py` generates paraphrases for the SNIPS dataset.

In the output file, paraphrases are represented as a list of phrases. For example:\
*(u'want a table', u'Book me a table', u'Make a reservation')*

# Visualize Essentia:
We implement a demo that takes as input a set of sentences and outputs (1) visualization of the graph representation of sentences and (2) candidate paraphrases mined from input.


## Run Essentia demo:
* Run `python demo/demo.py` at the root directory.

When running the demo, make sure to remove any empty line at the end of the input box.

# Run the baseline implementation for paraphrases mining:

We also built a baseline that uses constituency parsing tree to align sentences and create sentence graphs.
The baseline takes a list of sentences (with similar intents) as input, and outputs paraphrases.
More details can be found in the following paper by Bo Pang et al:\
[Syntax-based Alignment of Multiple Translations: Extracting Paraphrases and Generating New Sentences](https://www.aclweb.org/anthology/N03-1024.pdf)

## Run the baseline:
* Install [benepar 0.1.2](https://pypi.org/project/benepar/) and TensorFlow 1.13.1.
* Download benepar resource benepar_en2 using Python:\
  `>>> import nltk`\
  `>>> import benepar`\
  `>>> import benepar.download('benepar_en2')`
* Run the algorithm on a single file as follows:\
`python ./scripts/fsa_baseline/make_fsa_graph.py ../PATH/TO/FILE.txt alt`

This will print all the paraphrases on the screen.

* To run the code on a directory containing multiple files, navigate to ./scripts/fsa_baseline and run:\
`./exp.sh RELATIVE/PATH/TO/DATA/DIR alt` (e.g., use `./exp.sh ../../datasets alt` to run the code on SNIPS dataset.)

This command will compute all the paraphrases for all files in the specified directory and stores results under `./baseline_results`.
