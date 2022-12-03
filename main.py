import copy
import csv
import math
import numpy as np
import sys
# from keras.models import Sequential
# from keras.layers import Dense, Activation

if __name__ == "__main__":
    train_matrix = []
    test_matrix = []

    input_file = sys.argv[1]
    f = open(input_file, 'r')
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    for line in reader:
        train_matrix.append([eval(i) for i in line[1:]])
        print(train_matrix)
    
    f.close()
    # compute(train_matrix, test_matrix)