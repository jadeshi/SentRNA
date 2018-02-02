# SentRNA
A computational agent for inverse RNA folding, i.e. predicting a RNA sequence that folds into a given target structure.
* Combines neural-network prediction followed by adaptive walk refinement to propose valid sequences for complex targets
* Trained on human player solutions from the online RNA design game EteRNA (www.eternagame.org) with the goal of learning human-like design strategies.

## Author
Jade Shi

## Dependencies
Python: tensorflow, numpy, pickle

ViennaRNA (https://www.tbi.univie.ac.at/RNA/):
This program uses the RNAfold function from ViennaRNA 1.8.5, and the more updated RNAplot function from ViennaRNA 2.3.3 (both
included). Add the following directory paths to your $PATH environment variable:
1. /SentRNA/util/ViennaRNA-1.8.5/Progs (for RNAfold)
2. /SentRNA/util/ViennaRNA-1.8.5/Progs/rnaplot (for RNAplot)

## Benchmarks
Eterna100 (http://www.eternagame.org/web/blog/6136054/)

80 / 100 across 165 models trained (all located in models/trained_models)

## Usage instructions:
### 1. Training
To have SentRNA train a new model, supply the --mode train argument to SentRNA/run.py, along with any other relevant ones. An example command to train a new model can be found in the first line of SentRNA/test_pipeline.py file. This trains a model named "model". Feel free to use this as a template and play around with it.

The training data supplied to SentRNA/run.py via the --input_data argument should be a .pkl of a list, where each element is also a list and contains information about a single puzzle solution in the following format:
[puzzle name, dot-bracket structure, puzzle solution, locked bases].

Example: ['Hairpin', '(((....)))', 'CCCAAAAGGG', 'oooooooooo']

Alternatively, you can also give it a text file, with each line having the above information, separated by commas and arbitrary whitespace:

Example: Hairpin   ,    (((....)))   ,   CCCAAAAGGG   ,   oooooooooo

If you're using the text input format, you can also include just the name and the dot bracket. If you do this, the puzzle solution and locked bases parts will be set to placeholders (all A's and all o's respectively). This is for situations such as testing a model on a puzzle with no locked bases (NOT RECOMMENDED for training since then you'd just be training on dummy solutions)

The dataset used previously to train all SentRNA models is a list of player solutions spanning all single-state Progression puzzles and several Lab puzzles. This is located in data/train/eterna_complete_ss.pkl. Currently, given a dataset with n unique puzzles, SentRNA uses subsets of puzzles 1 to n-3 for training, n-2 for initial validation, and n for initial testing. Please keep this in mind if you want to train models using your own datasets.

Any trained model is saved in the automatically generated "test" directory. The layer sizes of the prediction neural network, along with the long-range features used in the model and the results of initial validation and testing can be found in the automatically generated "results" directory.

### 2. Testing
To test a trained model, use --mode test, and any other relevant arguments. An example of a testing command is found on the second line of the test_pipeline.py file. This takes a trained model named "model" and has it predict solutions for all 100 puzzles in the Eterna100. 

Once again, the puzzles to be tested should be passed via --input_data, in the same form as when training: a pkl of a list of lists of the form [puzzle name, dot-bracket, puzzle solution, locked bases]. A .pkl of all 100 Eterna100 puzzles is provided in data/test/eterna100.pkl as example test puzzles.

Note: in this mode, only bases corresponding to an 'x' in the locked bases string will be preserved from the player solution. All other positions will be assumed blank.

Example: ['Hairpin', '(((....)))', 'CCCAAAAGGG', 'xxxooooooo'] will be seen by SentRNA as CCCnnnnnn in test mode, where 'n' corresponds to an unfilled base.

You can also supply this information in text format, one puzzle per line, in the same format as you would training data. If you have a bunch of puzzles to test with no locked bases, you can just give the name and dot-bracket information.

A set of pre-trained models are also included in data/models/trained_models that you can also try out. To use one of these models, set the --test_model argument to the directory containing the model (see SentRNA/test_kyurem7.py for an example)

Results will be stored in the automatically generated "test_results" directory.


### 3. Refinement
To refine an initial model prediction using adaptive walk, use --mode refine and any other relevant arguments. An example is found in the third line of SentRNA/test_pipeline.py. This performs 300 adapative walk trajectories of length 30 on all 100 Eterna100 puzzles.

The data passed to --input_data should be the .pkl file generated via testing a model in the "test_results" directory. This is a list of lists, containing the following information for every puzzle tested:
[puzzle name, dot-bracket, predicted solution, structural accuracy]

where the structural accuracy is defined as the fraction of matching dot-bracket characters between the predicted and actual target structures. This number is only there to inform SentRNA of exact matches so that it doesn't need to refine the puzzle.

#### Example test + refinement pipeline
An example of a trained model and its performance on one of the Eterna100 puzzles, Kyurem 7, is included. Simply run SentRNA/test_kyurem7.py. The script will first predict an initial solution to Kyurem 7 using the trained model, followed by 100 rounds of refinement. The refinement efficacy outputted at the end represents the fraction of trials in which a valid solution to Kyurem 7 was sampled.
