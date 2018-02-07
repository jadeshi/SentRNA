import numpy as np
import os
from mutinf import *
from compute_mfe import *
import random
from copy import deepcopy
import draw_rna

def load_txt(filename):
    '''Loads a text file of puzzle information to be used by SentRNA. Each line should correspond to a puzzle and 
       should be ordered as:
       - Puzzle name
       - Dot bracket of target
       - Solution
       - Locked bases
       
       separated by commas.

       You can either supply all of this information (training), or just the name and dot bracket (testing). 
       If only name and dot bracket are provided, the solution and constraints will be set to a sequence of all A's and all o's respectively.'''
    data = open(filename).read().splitlines()
    output = []
    for i in data:
        puzzle_info = [j.strip() for j in i.split('    ')]
        if len(puzzle_info) == 2:
            puzzle_info += ['A' * len(puzzle_info[-1]), 'o' * len(puzzle_info[-1])]
        output.append(puzzle_info)
    return output

def compute_angle(triplet):
  '''Compute angle between three points.'''
  p1, p2, p3 = triplet
  v1 = (p1 - p2) / np.linalg.norm(p1 - p2)
  v2 = (p3 - p2) / np.linalg.norm(p3 - p2)
  if np.abs(np.abs(np.dot(v1, v2) / np.linalg.norm(v1)) - 1) > 1e-3:
    return np.arccos(np.dot(v1, v2) / np.linalg.norm(v1))    
  else:
    return 0.0

def compute_distance(v1, v2, axis=None):
  '''Compute Cartesian distance between two vectors v1 and v2. v1 can also be multidimensional.'''
  if axis:
    return np.sum((v1 - v2) ** 2, axis=axis) ** 0.5
  else:
    return np.sum((v1 - v2) ** 2) ** 0.5


def output_to_base(output):
  bases = ['A', 'U', 'C', 'G']
  loc = np.where(output == np.max(output))[0][0]
  return bases[loc]
  
  
def insert_base(seq, char, index):
  seq = list(seq)
  seq[index] = char
  return ''.join(seq)


def onehot(base):
  '''Generates NN inputs per feature "base"'''
  output = np.zeros(5)
  if base == 'A':  
    output[0] = 1
  elif base == 'U':
    output[1] = 1
  elif base == 'C':
    output[2] = 1
  elif base == 'G':
    output[3] = 1
  elif base == 'n':
    output[4] = 1
  return output


def generate_label(base):
  output = np.zeros(4)
  if base == 'A':
    output[0] = 1
  elif base == 'U':
    output[1] = 1
  elif base == 'C':
    output[2] = 1
  elif base == 'G':
    output[3] = 1
  return output


# Data parsing functions
def generate_2d(dot_bracket):
  '''Generate a 2d representation of a dot bracket string and puzzle solution.'''
  inputs = open('input', 'w')
  inputs.write('A' * len(dot_bracket) + '\n')
  inputs.write(dot_bracket)
  inputs.close()
  os.system('RNAplot < input')


def parse_ps(ps):
  '''Parses a RNAplot output file and returns:
     1. XY coordinates of all bases
     2. Stack pairs
     3. Sequence.'''
  write = 0
  seq_write = 0
  pairs_write = 0
  pairs = []
  output = []
  inputs = open(ps).readlines()
  for line in inputs:
    # Pairs write
    if pairs_write == 1 and '] def' in line:
      pairs_write = 0
    if pairs_write == 1:
      p1 = line.split()[0][1:]
      p2 = line.split()[1][:-1]
      pairs.append([int(p1) - 1, int(p2) - 1])
    if 'pairs [' in line:
      pairs_write = 1
    # Seq write
    if seq_write == 1:
      seq = line
      seq_write = 0
    if 'sequence' in line:
      seq_write = 1
    # Write coordinates
    if '] def' in line:
      write = 0
    if write == 1:
      x = line.split()[0][1:]
      y = line.split()[1][:-1]
      output.append([float(x), float(y)])
    if 'coor [' in line:
      write = 1
  return np.array(output), pairs, seq[:-2]

# Choose 1 draw_structure function. Both returns the Cartesian coordinates and list of base pairs given a structure.
# One uses RNAplot, while the other uses an in-house rendering that replicates exactly what's shown on the EteRNA GUI and
# creates more faithful representations of some puzzles such as Mutated chicken feet and Mat - Lot 2-2 B

# RNAplot
#def draw_structure(dot_bracket):
#    generate_2d(dot_bracket)
#    coords, pairs, _ = parse_ps('rna.ps')
#    return coords, pairs

# Eterna
def draw_structure(dot_bracket):
    coords, bonds = draw_rna.coords_as_list(dot_bracket)
    pairs = []
    for i in range(len(bonds)):
        if bonds[i] != -1:
            pair = [i, bonds[i]]
            if pair not in pairs and [bonds[i], i] not in pairs:
                pairs.append(pair)
    return coords, pairs

def sample_without_replacement(arr):
    random.shuffle(arr)
    return arr.pop()

# Featurizer utilities
def find_paired_base(index, pairs):
    for pair in pairs:
        if index in pair:
            for i in pair:
                if i != index:
                    return i, pair
    return None, None


# Featurizers   
def pair_features(coords, pairs, index, seq):
    paired, pair = find_paired_base(index, pairs)
    if paired != None:
        return onehot(seq[paired])
    else:
        return np.zeros(5)

    
def nearest_neighbor_features(coords, index, seq):
    if index == 0:
        nn = np.concatenate((np.zeros(5), onehot(seq[index+1])))
        angle = 0
    elif index == len(coords) - 1:
        nn = np.concatenate((onehot(seq[index-1]), np.zeros(5)))
        angle = 0
    else:
        nn = np.concatenate((onehot(seq[index-1]), onehot(seq[index+1])))
        angle = compute_angle(coords[index-1:index+2])
    return np.concatenate((nn, [angle]))


def generate_random_dataset(length, n_solutions):
    bases = ['A', 'U', 'C', 'G']
    solutions = []
    for i in range(n_solutions):
        fake_sol = []
        for j in range(length):
            fake_sol.append(random.choice(bases))
        solutions.append(''.join(fake_sol))
    return solutions


def generate_MI_features_list(progression, puzzle_name, threshold, force_add_features=True, random_append=4, MI_features_list=[]):
    MI_features_list = deepcopy(MI_features_list)
    solutions = []
    for puzzle in progression:
        if puzzle[0] == puzzle_name:
            dot_bracket = puzzle[-3]
            solution = puzzle[-2]
            solutions.append(solution)
    solutions += generate_random_dataset(len(dot_bracket), random_append)
    MI = mutual_information_matrix(solutions)
    coords, pairs = draw_structure(dot_bracket)
    counter = 0
    while counter < threshold:
        if np.amax(MI) == 0:
            return MI_features_list
        a, b = np.argwhere(MI == np.amax(MI))[0]
        MI[a][b] = 0
        distance = round(compute_distance(coords[a], coords[b]), 1)
        angle = round(compute_angle([coords[a-1], coords[a], coords[b]]), 1)
        feature = [distance, angle]
        if force_add_features:
            if feature not in MI_features_list:
                MI_features_list.append(feature)
                counter += 1
        else:
            if feature not in MI_features_list:
                MI_features_list.append(feature)
            counter += 1
    return MI_features_list


def mutual_information_features(coords, pairs, index, seq, MI_features_list, tolerance=1e-5):
    all_distances = np.array([round(compute_distance(coords[i], coords[index]), 1) for i in range(len(coords))])
    all_angles = np.array([round(compute_angle([coords[index - 1], coords[index], coords[i]]), 1) for i in range(len(coords))])
    MI_feature_vector = np.array([])
    for feature in MI_features_list:
        distance, angle = feature
        distance_similarities = np.where(np.abs(all_distances - distance) < tolerance)[0]
        angle_similarities = np.where(np.abs(all_angles - angle) < tolerance)[0]
        to_assign = np.intersect1d(distance_similarities, angle_similarities)
        if len(to_assign) > 0:
            MI_feature_vector = np.concatenate((MI_feature_vector, onehot(seq[to_assign[0]])))
        else:
            MI_feature_vector = np.concatenate((MI_feature_vector, np.zeros(5)))
    return MI_feature_vector


def featurize(coords, pairs, index, seq, MI_features_list):
    pair_comp = pair_features(coords, pairs, index, seq)
    nearest_neighbor_comp = nearest_neighbor_features(coords, index, seq)
    mutual_information_comp = mutual_information_features(coords, pairs, index, seq, MI_features_list)
    return np.concatenate((pair_comp, nearest_neighbor_comp, mutual_information_comp))


def prepare_single_base_environment(dot_bracket, seq, index, MI_features_list):
    """Generates local environment information for a single base position given a dot bracket and 
     sequence. Looping this function and passing the resulting data creates a dataset, which then,
     if passed to the NN evaluate function, simulates solving of a puzzle."""
    positions, pairs = draw_structure(dot_bracket)
    inputs = featurize(positions, pairs, index, seq, MI_features_list)
    label = generate_label(seq[index])
    return inputs, label


def prepare_prior_dataset(dot_bracket, solution, MI_features_list, fixed_bases=[None], shuffle=False, train_on_solved=False):
  """Takes a puzzle with a known solution, adds one base at a time to the unsolved empty sequence, 
     and returns local environment information of bases during this process. This simulates a situation
     in which the puzzle is being gradually solved by an Eterna player. If fixed bases is supplied, those bases
     will not be added to the local environment dataset. If train_on_solved=True, the process will be repeated 
     with all bases filled in at the start."""
  nn_inputs = []
  nn_labels = []
  order = range(len(dot_bracket))
  if shuffle:
    np.random.shuffle(order)
  if fixed_bases[0] != None:
    for fixed_base in fixed_bases:
      order.remove(fixed_base)
  for i in range(len(order)):
    if not train_on_solved:
      prior_solution = 'n' * len(dot_bracket)
    else:
      prior_solution = solution
    if fixed_bases[0] != None:
      for fixed_base in fixed_bases:
        prior_solution = insert_base(prior_solution, solution[fixed_base], fixed_base)
    bases_to_edit = order[:i+1]
    for base in bases_to_edit:
      prior_solution = insert_base(prior_solution, solution[base], base)
    inputs, label = prepare_single_base_environment(dot_bracket, prior_solution, order[i], MI_features_list)
    nn_inputs.append(inputs)
    nn_labels.append(label)
  return nn_inputs, nn_labels


def parse_progression_dataset(progression, puzzles, n_samples, MI_features_list, evaluate=False, shuffle=False, train_on_solved=False, **kwargs):
  '''Parses an Eterna dataset "progression" (in proper .pkl format) for a particular puzzle and returns a training or validation dataset for that puzzle'''
  training_dataset = []
  training_labels = []
  n_samples_list = []
  for puzzle_name in puzzles:
    counter = 0
    solutions = []
    puzzle_queue = []
    for puzzle in progression:
      if puzzle[0] == puzzle_name:
        puzzle_queue.append(puzzle)
    if type(n_samples) == list:
        n_samples_list == deepcopy(n_samples)
        n_samples = len(n_samples)
    for k in range(n_samples): 
      if len(n_samples_list) > 0:
          puzzle = puzzle_queue[n_samples_list[k]]
      else:
          puzzle = random.choice(puzzle_queue)
      dot_bracket = puzzle[-3]
      solution = puzzle[-2]
      constraints = puzzle[-1]
      fixed_bases = []
      for k in range(len(constraints)):
        if constraints[k] == 'x':
          fixed_bases.append(k)
      if evaluate:
        return dot_bracket, solution, fixed_bases
      if len(fixed_bases) == 0:
        nn_inputs, nn_labels = prepare_prior_dataset(dot_bracket, solution, MI_features_list, fixed_bases=[None], shuffle=shuffle)
      else:
        nn_inputs, nn_labels = prepare_prior_dataset(dot_bracket, solution, MI_features_list, fixed_bases, shuffle=shuffle)
      training_dataset += nn_inputs
      training_labels += nn_labels
      if train_on_solved:
        if len(fixed_bases) == 0:
          nn_inputs, nn_labels = prepare_prior_dataset(dot_bracket, solution, MI_features_list, fixed_bases=[None], shuffle=shuffle, train_on_solved=train_on_solved)
        else:
          nn_inputs, nn_labels = prepare_prior_dataset(dot_bracket, solution, MI_features_list, fixed_bases, shuffle=shuffle, train_on_solved=train_on_solved)
        training_dataset += nn_inputs
        training_labels += nn_labels
  rewards = np.ones(len(training_dataset))
  return np.array(training_dataset), np.array(training_labels), rewards


def create_hyperparameter_set(hyperparameters):
    '''Enumerate all unique combinations of hyperparameters given a list of lists of different hyperparameters.'''
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


def compute_MI_features(dataset, puzzle_names, puzzle_solution_count, min_n_solutions, features_per_puzzle, force_add_features, random_append):
    MI_features_list = []
    for puzzle in puzzle_names:
        if puzzle_solution_count[puzzle] >= min_n_solutions:
            print 'Generating MI features for %s'%(puzzle)
            MI_features_list = generate_MI_features_list(dataset, puzzle, features_per_puzzle, force_add_features, random_append, MI_features_list)
    return MI_features_list
