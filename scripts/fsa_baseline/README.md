# Run the baseline implementation for paraphrases mining:

We built a baseline that uses constituency parsing tree to align sentences and create sentence graphs.
The baseline takes a list of sentences (with similar intents) as input, and outputs paraphrases.
More details can be found in the following paper by Bo Pang et al:\
[Syntax-based Alignment of Multiple Translations: Extracting Paraphrases and Generating New Sentences](https://www.aclweb.org/anthology/N03-1024.pdf)

## Run the baseline:
* Install [benepar 0.1.2](https://pypi.org/project/benepar/) and TensorFlow 1.13.1.
* Run the algorithm on a single file as follows:\
`python ./scripts/fsa_baseline/make_fsa_graph.py ../PATH/TO/FILE.txt alt`

This will print all the paraphrases on the screen.

* To run the code on a directory containing multiple files, navigate to ./scripts/fsa_baseline and run:\
`./exp.sh RELATIVE/PATH/TO/DATA/DIR alt` (e.g., use `./exp.sh ../../datasets alt` to run the code on SNIPS dataset.)

This command will compute all the paraphrases for all files in the specified directory and stores results under `./baseline_results`.
