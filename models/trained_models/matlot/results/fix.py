import os
import pickle

for i in os.listdir(os.getcwd()):
    if 'MI_features_list' in i:
        a = pickle.load(open(i))
        for j in range(len(a)):
            a[j] = a[j][0]
        pickle.dump(a, open(i, 'w'))
