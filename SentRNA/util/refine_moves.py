import sys
import numpy as np
import os
from copy import deepcopy
from compute_mfe import *
import random

def find_stacks(pairs):
  '''Returns a list of of lists, where each list contains a set of stack base pairs.'''
  stacks = []
  if len(pairs) == 0:
    return stacks
  stack = [pairs[0]]
  for i in range(1,len(pairs)):
    if np.all(np.abs(np.array(pairs[i]) - np.array(pairs[i-1])) == np.ones(2)):
      stack.append(pairs[i])
      if i == len(pairs) - 1:
        stacks.append(stack)
        return stacks
    else:
      stacks.append(stack)
      stack = [pairs[i]]
  return stacks

def insert_base(seq, index, char):
  seq = list(seq)
  seq[index] = char
  return ''.join(seq)

# Moveset        
class BoostMove():
    def __init__(self, mutations=False):
        if mutations:
            self.mutations = mutations
        else:
            self.mutations = [['U', 'U'], ['G', 'A'], ['A', 'G']]

    def mutate(self, seq, pair, mutation):
        seq = insert_base(seq, pair[0], mutation[0])
        seq = insert_base(seq, pair[1], mutation[1])
        return seq

    def apply(self, dot_bracket, current_solution):
        accuracy, true_pairs, bad_pairs, missing_pairs = refine_check_answer(dot_bracket, current_solution)
        stacks = find_stacks(true_pairs)
        stack_indices = np.unique(np.array([i for stack in stacks for pair in stack for i in pair]).astype(int))
        mutable_bases = []
        for stack in stacks:
            mut1 = [stack[0][0] - 1, stack[0][1] + 1]
            mut2 = [stack[-1][0] + 1, stack[-1][1] - 1]
            if -1 not in mut1 and len(current_solution) not in mut1 and mut1[0] not in stack_indices and mut1[1] not in stack_indices:
                mutable_bases.append(mut1)
            if -1 not in mut2 and len(current_solution) not in mut2 and mut2[0] not in stack_indices and mut2[1] not in stack_indices:
                mutable_bases.append(mut2)
        if len(mutable_bases) == 0:
            return [accuracy, bad_pairs, missing_pairs], current_solution
        index = np.random.choice(range(len(mutable_bases)))
        pair = mutable_bases[index]
        mutation = random.choice(self.mutations)
        mod_solution = deepcopy(current_solution)
        mod_solution = self.mutate(mod_solution, pair, mutation)
        new_accuracy, _, new_bad_pairs, new_missing_pairs = refine_check_answer(dot_bracket, mod_solution)
        return [new_accuracy, new_bad_pairs, new_missing_pairs], mod_solution

class MissingPairsMove():
    '''Mutate unpaired bases that should be paired to a suitable pair.'''
    def __init__(self, mutations=False):
        if mutations:
            self.mutations = mutations
        else:
            self.mutations = [['G', 'U'], ['U', 'G'], ['G', 'C'], ['C', 'G'], ['A', 'U'], ['U', 'A']]
     
    def mutate(self, seq, pair, mutation):
        seq = insert_base(seq, pair[0], mutation[0])
        seq = insert_base(seq, pair[1], mutation[1])
        return seq

    def apply(self, dot_bracket, current_solution):
        accuracy, true_pairs, bad_pairs, missing_pairs = refine_check_answer(dot_bracket, current_solution)
        scores = []
        actions = []
        if len(missing_pairs) == 0:
            return [accuracy, bad_pairs, missing_pairs], current_solution
        pair = random.choice(missing_pairs)
        mutation = random.choice(self.mutations)
        mod_solution = deepcopy(current_solution)
        mod_solution = self.mutate(mod_solution, pair, mutation)
        new_accuracy, _, new_bad_pairs, new_missing_pairs = refine_check_answer(dot_bracket, mod_solution)
        return [new_accuracy, new_bad_pairs, new_missing_pairs], mod_solution


class BadPairsMove():
    '''Mutate bad pairs that should not exist away to something not amenable to pairing'''
    def __init__(self, mutations=False):
        if mutations:
            self.unpaired_mutations = mutations
        else:
            self.unpaired_mutations = [['A', 'A'], ['U', 'U'], ['C', 'C'], ['G', 'G'], \
                                   ['A', 'C'], ['C', 'A'], ['A', 'G'], ['G', 'A'], \
                                   ['U', 'C'], ['C', 'U']]
        
    def mutate(self, seq, pair, mutation):
        seq = insert_base(seq, pair[0], mutation[0])
        seq = insert_base(seq, pair[1], mutation[1])
        return seq

    def apply(self, dot_bracket, current_solution):
        accuracy, true_pairs, bad_pairs, missing_pairs = refine_check_answer(dot_bracket, current_solution)
        scores = []
        actions = []
        if len(bad_pairs) == 0:
            return [accuracy, bad_pairs, missing_pairs], current_solution 
        pair = random.choice(bad_pairs)
        mutation = random.choice(self.unpaired_mutations)
        mod_solution = deepcopy(current_solution)
        mod_solution = self.mutate(mod_solution, pair, mutation)
        new_accuracy, _, new_bad_pairs, new_missing_pairs = refine_check_answer(dot_bracket, mod_solution)
        return [new_accuracy, new_bad_pairs, new_missing_pairs], mod_solution


class GoodPairsMove():
    '''If pairing is already good, either mutate to another set of pairing bases or swap them.'''
    def __init__(self, mutations=False):
        if mutations:
            self.pair_mutations = mutations
        else:
            self.pair_mutations = [['G', 'U'], ['U', 'G'], ['G', 'C'], ['C', 'G'], ['A', 'U'], ['U', 'A']]
    
    def swap(self, seq, pair):
        base_1, base_2 = seq[pair[0]], seq[pair[1]]
        seq = insert_base(seq, pair[1], base_1)
        seq = insert_base(seq, pair[0], base_2)
        return seq

    def mutate(self, seq, pair, mutation):
        seq = insert_base(seq, pair[0], mutation[0])
        seq = insert_base(seq, pair[1], mutation[1])
        return seq

    def apply(self, dot_bracket, current_solution):
        accuracy, true_pairs, bad_pairs, missing_pairs = refine_check_answer(dot_bracket, current_solution)
        scores = []
        actions = []
        good_pairs = []
        for i in true_pairs:
            if i not in bad_pairs and i not in missing_pairs:
                good_pairs.append(i)
        if len(good_pairs) == 0:
            return [accuracy, bad_pairs, missing_pairs], current_solution
        pair = random.choice(good_pairs)
        mutation = random.choice(self.pair_mutations)
        mod_solution = deepcopy(current_solution)
        mod_solution = self.mutate(mod_solution, pair, mutation)
        new_accuracy, _, new_bad_pairs, new_missing_pairs = refine_check_answer(dot_bracket, mod_solution)
        return [new_accuracy, new_bad_pairs, new_missing_pairs], mod_solution

