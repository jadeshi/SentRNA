import numpy as np
import pickle
from compute_mfe import *

def count_bases(data, position):
    base_counts = {'A':1e-10,'U':1e-10,'C':1e-10,'G':1e-10}
    total_counts = 0
    for solution in data:
        base_counts[solution[position]] += 1.0
        total_counts += 1.0
    for key in base_counts.keys():
        base_counts[key] /= total_counts
    return base_counts
        

def ignore_bases(data):
    A_counts = np.zeros(len(data[0]))
    for i in range(len(data[0])):
        for puzzle in data:
            if puzzle[i] == 'A':
                A_counts[i] += 1
    A_counts /= len(data)
    return np.where(A_counts > 0.8)[0]


def positional_mutual_information(index, data):
    '''Computes the mutual information between base at "index" with all other
    bases in the puzzle.'''
    sols = {'A_sols':[], 'U_sols':[], 'C_sols':[], 'G_sols':[]}
    for solution in data:
        sols['%s_sols'%(solution[index])].append(solution)
    probs = {'unconditioned':None, 'A_conditioned':None,'U_conditioned':None,
             'C_conditioned':None, 'G_conditioned':None}
    marginal_probs = {'A':0,'U':0,'C':0,'G':0} # Note, marginal is P(X)
    for key in marginal_probs.keys():
        marginal_probs[key] = len(sols['%s_sols'%(key)]) / float(len(data))
    MIs = np.zeros(len(data[0]))
    positions = range(len(data[0]))
    positions.remove(index)
    for position in positions:
        # Unconditioned probabilities
        probs['unconditioned'] = count_bases(data, position)
        # Conditioned probabilities
        for i in ['A', 'U', 'C', 'G']:
            if len(sols['%s_sols'%(i)]) > 0:
                # Note, probs[X_conditioned] is P(Y|X)
                probs['%s_conditioned'%(i)] = count_bases(sols['%s_sols'%(i)], position)
            else:
                probs['%s_conditioned'%(i)] = probs['unconditioned'] # If a certain base does not exist in a position, consider that part of the MI 0
        MI = 0
        for i in ['A', 'U', 'C', 'G']:
            for j in ['A','U','C','G']:
                MI += probs['%s_conditioned'%(i)][j] * marginal_probs[i] * np.log(probs['%s_conditioned'%(i)][j] / probs['unconditioned'][j])
        MIs[position] = MI
    return MIs

def mutual_information_matrix(data, ignore_indices):
    MIs = []
    for index in range(len(data[0])):
        MIs.append(positional_mutual_information(index, data))
    MIs = np.array(MIs)
    print ignore_indices
    if ignore_indices != []:
        MIs[:,ignore_indices] = 0
        MIs[ignore_indices,:] = 0
    return MIs
