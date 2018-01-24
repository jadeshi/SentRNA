import numpy as np
import os
import pickle
import feedforward as rl
from featurize_util import *
from compute_mfe import *


def create_hyperparameter_set(hyperparameters):
    if len(hyperparameters) == 2:
        set_1, set_2 = hyperparameters[0], hyperparameters[1]
        combinations = []
        for i in set_1:
            for j in set_2:
                if type(i) != list:
                    i = [i]
                if type(j) != list:
                    j = [j]
                combination = i + j
                combinations.append(combination)
        return combinations
    else:
        return create_hyperparameter_set([hyperparameters[0], create_hyperparameter_set(hyperparameters[1:])])

def train(results_path, nn_training_dataset, nn_val_dataset, nn_test_dataset, layer_sizes, nb_epochs, mini_epoch):
    model = rl.TensorflowClassifierModel(layer_sizes=layer_sizes)
    dot_bracket, seq, fixed_bases = nn_val_dataset
    val_solutions = []
    val_accuracies = []
    if results_path not in os.listdir('test'):
        os.mkdir('test/%s'%(results_path))
    for i in range(nb_epochs / mini_epoch):
        save_path = '%s_mini-epoch_%d.ckpt'%(results_path, i)
        restore_path = '%s_mini-epoch_%d.ckpt'%(results_path, i - 1)
        print 'Mini epoch %d'%(i)
        val_model = 'test/%s-%d'%(save_path, mini_epoch + 1)
        if '%s-%d.meta'%(restore_path, mini_epoch + 1) in os.listdir('test'):
            checkpoint = 'test/%s-%d'%(restore_path, mini_epoch + 1)
        else:
            checkpoint = None
        model.fit(nn_training_dataset, loss_thresh=1e-9, nb_epochs=mini_epoch, save_path=save_path, checkpoint=checkpoint)
        # Validation
        val_solution, rec = model.evaluate(dot_bracket, seq, fixed_bases, layer_sizes, MI_features_list, val_model)
        val_accuracy = check_answer(val_solution, dot_bracket)
        # Validation after refinement
        val_solution2, rec2 = model.evaluate(dot_bracket, val_solution, fixed_bases, layer_sizes, MI_features_list, val_model, refine=True)
        val_accuracy2 = check_answer(val_solution2, dot_bracket)
        if val_accuracy2 > val_accuracy:
            val_accuracy = val_accuracy2
            val_solution = val_solution2
        val_solutions.append(val_solution)
        val_accuracies.append(val_accuracy)
    pickle.dump(rec, open('results/val_inputs.%s.pkl'%(results_path), 'w'))
    val_accuracies = np.array(val_accuracies)
    best_model = np.where(np.array(val_accuracies) == np.max(val_accuracies))[0][0]
    print 'Maximum validation accuracy is %f after mini-epoch %d'%(val_accuracies[best_model], best_model)
    # Test
    test_model = 'test/%s_mini-epoch_%d.ckpt-%d'%(results_path, best_model, mini_epoch + 1)
    dot_bracket, seq, fixed_bases = nn_test_dataset
    test_solution, rec = model.evaluate(dot_bracket, seq, fixed_bases, layer_sizes, MI_features_list, test_model)
    test_accuracy = check_answer(test_solution, dot_bracket)
    test_solution2, rec2 = model.evaluate(dot_bracket, test_solution, fixed_bases, layer_sizes, MI_features_list, test_model, refine=True)
    test_accuracy2 = check_answer(test_solution2, dot_bracket)
    if test_accuracy2 > test_accuracy:
        test_solution = test_solution2
        test_accuracy = test_accuracy2
    print 'Test solution'
    print test_solution, test_accuracy
    pickle.dump([best_model, val_solutions[best_model], val_accuracies[best_model], test_solution, test_accuracy], open('results/%s.pkl'%(results_path), 'w'))
    os.system('mv test/%s_mini-epoch_%d* test/%s'%(results_path, best_model, results_path))
    os.system('rm test/*')
    return 0

if __name__ == '__main__':
    eterna_progression = pickle.load(open('../../eterna_complete_ss.pkl'))
    unique_puzzles = pickle.load(open('../../unique_puzzles_complete_ss.pkl'))
    puzzle_solution_count = pickle.load(open('../../puzzle_solution_count.pkl'))
    eterna100 = pickle.load(open('../../eterna100.pkl'))
    append_puzzles = open('good_2-2_puzzles_names').read().splitlines()
    for puzzle in range(len(unique_puzzles) - 1):
        if unique_puzzles[puzzle] == '"TR1B1 - Shape 3"':
            possible_train_puzzles = unique_puzzles[:puzzle]
            val_puzzle = unique_puzzles[puzzle]
            test_puzzle = 'Mat - Lot 2-2 B'
            #test_puzzle = unique_puzzles[puzzle+2] # There's one control puzzle near the end that's useless
    # Hyperparameter set for training  
    n_samples_list = [1]
    shuffle_list = [False]
    random_append_list = range(0,1)
    threshold_list = range(1,2)
    force_add_features_list = [True]
    hyperparameter_set = [n_samples_list, force_add_features_list, threshold_list, shuffle_list, random_append_list]
    hyperparam_list = create_hyperparameter_set(hyperparameter_set)
    for params in hyperparam_list:
        n_samples, force_add_features, threshold, shuffle, random_append = params
        # Generate MI features
        MI_features_list = []
        puzzle_count_threshold = 50
        for puzzle in possible_train_puzzles:
            if puzzle_solution_count[puzzle] > puzzle_count_threshold:
                print 'Generating MI features for %s'%(puzzle)
                MI_features_list = generate_MI_features_list(eterna_progression, puzzle, threshold, force_add_features, MI_features_list)
        #MI_features_list = rank_MI_by_val(MI_features_list)
        #MI_features_master = deepcopy(MI_features_list)
        n_train_puzzles = 50
        n_trials = 100
        #to_train_on = random.sample(range(len(possible_train_puzzles)), n_train_puzzles)
        #train_puzzles = [possible_train_puzzles[i] for i in to_train_on]
        for trial in range(n_trials):
            #np.random.shuffle(MI_features_master) 
            #for MI_limit_ in range(len(MI_features_list)):
            if True:
                #to_train_on = random.sample(range(len(possible_train_puzzles)), n_train_puzzles)
                #train_puzzles = [possible_train_puzzles[i] for i in to_train_on]
                train_puzzles = append_puzzles
                results_path = 'MI-%d_trial%d'%(len(MI_features_list), trial)
                #results_path = 'MI-%d_trial%d'%(MI_limit_, trial)
                print results_path
                if '%s.pkl'%(results_path) in os.listdir('results'):
                    print 'Skipping %s'%(results_path)
                    continue
                #MI_features_list = MI_features_master[:MI_limit_]
                training_info, inputs, labels, rewards = parse_progression_dataset(eterna_progression, train_puzzles, n_samples, MI_features_list, evaluate=False, train_on_solved=True)
                pickle.dump(training_info, open('results/training_puzzles_%s.pkl'%(results_path), 'w'))
                nn_training_dataset = [inputs, labels, rewards, None]
                nn_val_dataset = parse_progression_dataset(eterna_progression, [val_puzzle], 1, MI_features_list, evaluate=True)
                nn_test_dataset = parse_progression_dataset(eterna100, [test_puzzle], 1, MI_features_list, evaluate=True)
                n_layers = 3
                input_size = len(nn_training_dataset[0][0])
                hidden_size = 100
                layer_sizes = [input_size] + [hidden_size] * n_layers + [4]
                nb_epochs = 1000
                mini_epoch = 100
                # Save MI feature vector because we can
                pickle.dump(MI_features_list, open('results/MI_features_list.%s.pkl'%(results_path), 'w')) # Save MI features
                train(results_path, nn_training_dataset, nn_val_dataset, nn_test_dataset, layer_sizes, nb_epochs, mini_epoch)

