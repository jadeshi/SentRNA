import pickle
import os

os.system("python run.py --mode test --input_data ../data/test/eterna100.pkl --test_model val_test_n_samples=2_force_add_features=False_threshold=2_shuffle=True_random_append=0_train_on_solved --results_path val_test_n_samples=2_force_add_features=False_threshold=2_shuffle=True_random_append=0_train_on_solved.pkl --test_puzzle_name 'Kyurem 7'")

for i in range(100):
    os.system("python run.py --mode refine --input_data test_results/val_test_n_samples=2_force_add_features=False_threshold=2_shuffle=True_random_append=0_train_on_solved.pkl --results_path kyurem7_%d.pkl --refine_puzzle_name 'Kyurem 7'"%(i))

eff = 0
for i in os.listdir('refined'):
    results = pickle.load(open('refined/%s'%(i)))
    if results[-1][-1] == 1:
        eff += 1
print 'Solution efficacy is %f'%(float(eff) / len(os.listdir('refined')))
