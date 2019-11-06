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
* Copy the *monolingualWordAligner* directory in the repository into `./scripts/essentia/`.
* We reimplement the word aligner to make it run with Python 3. Copy all files under `./scripts/essentia/word_aligner` into `./scripts/essentia/monolingualWordAligner`.


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

---

We also implement a baseline algorithm to compare Essentia to. The baseline can be found at `./scripts/fsa_baseline`.