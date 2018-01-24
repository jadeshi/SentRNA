# SentRNA
A computational agent for inverse RNA folding, i.e. predicting a RNA sequence that folds into a given target structure.
* Combines neural-network prediction followed by adaptive walk refinement to propose valid sequences for complex targets
* Trained on human player solutions from the online RNA design game EteRNA (www.eternagame.org)

## Author
Jade Shi

## Dependencies
Python: tensorflow, numpy, pickle

ViennaRNA 1.8.5: Add the included /SentRNA/util/ViennaRNA-1.8.5/Progs directory to the $PATH environment variable

## Benchmarks:
Eterna100 (http://www.eternagame.org/web/blog/6136054/)

80 / 100

## Usage instructions:
### 1. Training
An example template for the full process of training, validation, testing, and refinement is included: /SentRNA/test_pipeline.sh. Training a model is handled by the first command in the file.

The dataset used to train the model is a custom-compiled list of player solutions across 724 unique target structures (puzzles). Currently, by design only puzzles 1 to 722 are used for training, whereas puzzle 722 and 724 are reserved for initial validation and testing.

The trained model is saved in the automatically generated "test" directory.

### 2. Testing
Testing a model is handled by the second command in the /SentRNA/test_pipeline.sh file. This takes the trained model and has it predict solutions for all 100 puzzles in the Eterna100 given their target structur
es. Results will be stored in the automatically generated "test_results" directory.

### 3. Refinement
Refinement is handled by the third command in the /SentRNA/test_pipeline.sh file. This will take the initially predicted solution stored in "test_results" for each puzzle and attempt to refine it using an adaptive walk algorithm. Results will be saved in the automatically generated "refined" directory.

### Example refinement
An example of a trained model and its performance on one of the Eterna100 puzzles, Kyurem 7, is included. To use, copy everything from the /SentRNA/kyurem7_trained directory into /SentRNA and run "test_kyurem7.py". The script will first predict an initial solution to Kyurem 7 using the trained model, followed by 100 rounds of refinement. The refinement efficacy outputted at the end represents the fraction of trials i
n which a valid solution to Kyurem 7 was sampled.
