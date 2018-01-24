import numpy as np
import os
from subprocess import Popen, PIPE, STDOUT
import re

def seq_to_struct(pred_solution):
    '''Returns predicted structure and energy given a sequence.'''
    p = Popen(['RNAfold'], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
    output = p.communicate(input=pred_solution)[0]
    pred_struct = re.split('\s+| \(?\s?', output)[1]
    return pred_struct

def check_answer(pred_solution, dot_bracket):
    '''Checks percentage of dot-bracket agreement between a predicted structure and target structure.'''
    pred_struct = seq_to_struct(pred_solution)
    correct = 0
    for i in range(len(pred_struct)):
        if pred_struct[i] == dot_bracket[i]:
            correct += 1
    return correct / float(len(dot_bracket))

def bracket_to_bonds(structure):
    bonds = [None]*len(structure)
    opening = []
    for i,c in enumerate(structure):
        if c == '(':
            opening.append(i)
        elif c == ')':
            j = opening.pop()
            bonds[i] = j
            bonds[j] = i
    reshaped_bonds = []
    for i in range(len(bonds)):
        if bonds[i] != None:
            pair = [i, bonds[i]]
            if pair not in reshaped_bonds and [bonds[i], i] not in reshaped_bonds:
                reshaped_bonds.append(pair)
    return reshaped_bonds

def refine_check_answer(dot_bracket, pred_solution):
  pred_struct = seq_to_struct(pred_solution)
  correct = 0
  for i in range(len(pred_struct)):
    if pred_struct[i] == dot_bracket[i]:
      correct += 1
  # Compute long-range erroneous pairs
  pred_pairs = bracket_to_bonds(pred_struct)
  true_pairs = bracket_to_bonds(dot_bracket)
  bad_pairs = []
  for i in pred_pairs:
    if i not in true_pairs:
      bad_pairs.append(i)
  missing_pairs = []
  for i in true_pairs:
    if i not in pred_pairs:
      missing_pairs.append(i)
  return correct / float(len(dot_bracket)), true_pairs, bad_pairs, missing_pairs
  
