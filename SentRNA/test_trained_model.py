import pickle
import os

############ PASTE PARAMS HERE ######################
path = '../models/trained_models/batch1/test/trial-14_MI-20'
puzzle_name = '"1,2,3and4bulges"'
renderer = 'rnaplot'
moveset = "[GoodPairsMove(), MissingPairsMove()]"
n_trajs = 300
############ PASTE PARAMS HERE ######################


os.system("python run.py --mode test --renderer %s --input_data ../data/test/eterna100.pkl --test_model %s --results_path out.pkl --test_puzzle_name '%s'"%(renderer, path, puzzle_name))

for i in range(100):
    os.system("python run.py --mode refine --input_data test_results/out.pkl --results_path refined_%d.pkl --refine_puzzle_name '%s' --move_set \"%s\" --n_trajs %d"%(i, puzzle_name, moveset, n_trajs))

eff = 0
for i in os.listdir('refined'):
    results = pickle.load(open('refined/%s'%(i)))
    if results[-1][-1] == 1:
        eff += 1
print 'Solution efficacy is %f'%(float(eff) / len(os.listdir('refined')))
