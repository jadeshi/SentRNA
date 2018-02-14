import os
import pickle

# Train test
os.system('python run.py --mode train -input_data ../data/train/eterna_complete_ss.pkl --results_path model --n_long_range_features 20')
# Test test
os.system('python run.py --mode test --input_data ../data/test/eterna100.pkl --test_model test/model --results_path model_tested.pkl')
# Refine test
os.system('python run.py --mode refine --input_data test_results/model_tested.pkl --results_path model_refined.pkl')

results = pickle.load(open('refined/model_refined.pkl'))
success = 0
for i in results:
    if i[-1] == 1:
        success += 1
print 'Solved %f of Eterna100'%(success / float(len(results)))
