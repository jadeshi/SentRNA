#!/bin/bash

# Train test
python run.py --mode train --input_data ../data/train/eterna_complete_ss.pkl --puzzle_names ../data/train/eterna_complete_ss.puzzle_names.pkl --puzzle_solution_count ../data/train/eterna_complete_ss.puzzle_solution_count.pkl --results_path model --n_long_range_features 20
# Test test
python run.py --mode test --input_data ../data/test/eterna100.pkl --test_model model --results_path model_tested
# Refine test
python run.py --mode refine --input_data test_results/model_tested.pkl --results_path model_refined

