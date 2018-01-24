import numpy as np
import os
import pickle
import feedforward as rl
from featurize_util import *
from compute_mfe import *
from extract_hyperparameter_set import *

if __name__ == '__main__':
  eterna_progression = pickle.load(open('../../../eterna_complete_ss.pkl'))
  unique_puzzles = pickle.load(open('../../../unique_puzzles_complete_ss.pkl'))
  for puzzle in range(len(unique_puzzles) - 1):
      if unique_puzzles[puzzle] == '"TR1B1 - Shape 3"':
          val_puzzle = unique_puzzles[puzzle]
          test_puzzle = unique_puzzles[puzzle+2] # There's one control puzzle near the end that's useless
  n_samples = 1
  MI_features_list = []
  n_trials = 100
  append_set = pickle.load(open('3loop_sols.pkl'))
  append_names = [i[0] for i in append_set]
  for trial in range(n_trials):
          results_path = '3loop_%d'%(trial)
          if results_path not in os.listdir('test'):
              os.mkdir('test/%s'%(results_path))
          append_inputs, append_labels, append_rewards = parse_progression_dataset(append_set, append_names, n_samples, MI_features_list, evaluate=False, shuffle=False, train_on_solved=True)
          nn_training_dataset = [append_inputs, append_labels, append_rewards, None]
          # Model params
          n_layers = 3
          nb_epochs = 1000
          mini_epoch = 100
          input_size = len(nn_training_dataset[0][0])
          hidden_size = 100
          layer_sizes = [input_size] + [hidden_size] * n_layers + [4]
          pickle.dump(MI_features_list, open('results/%s.pkl'%(results_path), 'w')) # Save MI features
          val_accuracies = []
          val_solutions = []
          model = rl.TensorflowClassifierModel(layer_sizes=layer_sizes)
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
              dot_bracket, seq, fixed_bases = parse_progression_dataset(eterna_progression, [val_puzzle], 1, MI_features_list, evaluate=True)
              val_solution, rec = model.evaluate(dot_bracket, seq, fixed_bases, layer_sizes, MI_features_list, val_model)
              val_accuracy = check_answer(val_solution, dot_bracket)
              # Refine val
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
          test_model = 'test/%s_mini-epoch_%d.ckpt-%d'%(results_path, best_model, mini_epoch + 1)
          dot_bracket, seq, fixed_bases = parse_progression_dataset(eterna_progression, [test_puzzle], 1, MI_features_list, evaluate=True)
          test_solution, rec = model.evaluate(dot_bracket, seq, fixed_bases, layer_sizes, MI_features_list, test_model)
          test_accuracy = check_answer(test_solution, dot_bracket)
          test_solution2, rec2 = model.evaluate(dot_bracket, test_solution, fixed_bases, layer_sizes, MI_features_list, test_model, refine=True)
          test_accuracy2 = check_answer(test_solution2, dot_bracket)
          if test_accuracy2 > test_accuracy:
              test_solution = test_solution2
              test_accuracy = test_accuracy2
          print 'Test solution'
          print test_solution, test_accuracy
          pickle.dump([best_model, val_solutions[best_model], val_accuracies[best_model], test_solution, test_accuracy], open('results/output.%s.pkl'%(results_path), 'w'))
          os.system('mv test/%s_mini-epoch_%d* test/%s'%(results_path, best_model, results_path))
          os.system('rm test/*')

