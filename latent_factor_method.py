import copy
import csv
import math
import numpy as np
import sys
import random
import os
import re

u = []
v = []

def calculate_rmse(temp_train_matrix, train_matrix):
    rmse = 0
    for i in range(len(temp_train_matrix)):
        for j in range(len(temp_train_matrix[0])):
            rmse += (train_matrix[i][j] - temp_train_matrix[i][j]) ** 2
    rmse /= len(train_matrix)
    rmse = math.sqrt(rmse)
    return rmse

def compute(latent_factor_size, train_matrix):
    permutation_u = []
    for i in range(len(train_matrix)):
        for k in range(latent_factor_size):
            permutation_u.append((i, k))
    permutation_v = []
    for k in range(latent_factor_size):
        for j in range(len(train_matrix[0])):
            permutation_v.append((k, j))
    random.shuffle(permutation_u)
    random.shuffle(permutation_v)

    # initialise u and v
    u = [[0] * latent_factor_size for _ in range(len(train_matrix))] # u: i*k
    v = [[0] * len(train_matrix[0]) for _ in range(latent_factor_size)] # v: k*j
    for i in range(len(train_matrix)):
        for j in range(len(train_matrix[0])):
            if train_matrix[i][j] != 0:
                for k in range(latent_factor_size):
                    u[i][k] = math.sqrt(train_matrix[i][j]/latent_factor_size)
                    v[k][j] = math.sqrt(train_matrix[i][j]/latent_factor_size)
                    # u[i][k] = round(random.random() * 3, 2)
                    # v[k][j] = round(random.random() * 3, 2)
    
    temp_train_matrix = []
    for i in range(len(train_matrix)):
        temp_train_matrix.append([])
        for j in range(len(train_matrix[0])):
            temp_value = 0
            for k in range(latent_factor_size):
                temp_value += u[i][k] * v[k][j]
            temp_train_matrix[i].append(temp_value)

    rmse = calculate_rmse(temp_train_matrix, train_matrix)
    previous_rmse = rmse + 10
    print(rmse)
    while (abs(previous_rmse - rmse) > 0.005):
        for (row, col) in permutation_u:
            top_sum = 0
            bottom_sum = 0
            for j in range(len(train_matrix[0])):
                if train_matrix[row][j] != 0:
                    v_sj = v[col][j]
                    m_rj = train_matrix[row][j]
                    temp = 0
                    for k in range(latent_factor_size):
                        if k != col:
                            temp += u[row][k] * v[k][j]
                    top_sum += v_sj * (m_rj - temp)
            for j in range(len(train_matrix[0])):
                if train_matrix[row][j] != 0:
                    v_ir = v[col][j]
                    bottom_sum += v_ir ** 2
            u[row][col] = top_sum / bottom_sum
                        
        for (row, col) in permutation_v:
            top_sum = 0
            bottom_sum = 0
            for i in range(len(train_matrix)):
                if train_matrix[i][col] != 0:
                    u_ir = u[i][row]
                    m_is = train_matrix[i][col]
                    temp = 0
                    for k in range(latent_factor_size):
                        if k != row:
                            temp += u[i][k] * v[k][col]
                    top_sum += u_ir * (m_is - temp)
            for i in range(len(train_matrix)):
                if train_matrix[i][col] != 0:
                    v_ir = u[i][row]
                    bottom_sum += v_ir ** 2
            
            v[row][col] = top_sum / bottom_sum
    
        # print("------u------")
        # for x in u:
        #     print(x)
        # print("------new u------")
        # for x in new_u:
        #     print(x)

        random.shuffle(permutation_u)
        random.shuffle(permutation_v)

        temp_train_matrix = []
        for i in range(len(train_matrix)):
            temp_train_matrix.append([])
            for j in range(len(train_matrix[0])):
                temp_value = 0
                for k in range(latent_factor_size):
                    temp_value += u[i][k] * v[k][j]
                temp_train_matrix[i].append(temp_value)

        previous_rmse = rmse
        rmse = calculate_rmse(temp_train_matrix, train_matrix)
    train_matrix = copy.deepcopy(temp_train_matrix)
    

def write_to_output_file(output_file_name, matrix):
    f = open(output_file_name, 'w')
    writer = csv.writer(f)
    for line in matrix:
      writer.writerow(line)
    f.close()

if __name__ == "__main__":
    train_matrix = []
    test_matrix = []
    latent_factor_size = 20

    file_name = "google_review_ratings_small.csv" # the original file
    train_dataset_file_name = "train_data_before_prediction.csv"

    f = open(file_name, 'r')
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    for line in reader:
        line = [re.sub('\s+', '', s) for s in line]
        test_matrix.append([eval(i) for i in line[1:]])
        train_matrix.append([eval(i) for i in line[1:]])
    f.close()

    # Output the test_data.csv file
    write_to_output_file("test_data.csv", test_matrix)

    if os.path.exists(train_dataset_file_name):
        print("train_data_before_prediction.csv file exists, use this dataset to train.")
        train_matrix = []
        f = open(train_dataset_file_name, 'r')
        reader = csv.reader(f)
        for line in reader:
            train_matrix.append([eval(i) for i in line])
        f.close()
    else:
        # Create train data if no train data created
        for r in range(len(train_matrix)):
            for c in range(len(train_matrix[0])):
                if random.random() > 0.3:
                    train_matrix[r][c] = 0
        # Output the train_data_before_prediction.csv file
        write_to_output_file("train_data_before_prediction.csv", train_matrix)

    compute(latent_factor_size, train_matrix)
    # Output the latent_factor_train_data_after_prediction.csv file
    write_to_output_file("latent_factor_train_data_after_prediction.csv", train_matrix)

    rmse = 0
    for i in range(len(test_matrix)):
        for j in range(len(test_matrix[0])):
            rmse += (train_matrix[i][j] - test_matrix[i][j]) ** 2
        rmse /= len(train_matrix)
        rmse = math.sqrt(rmse)
    print("RMSE is {} after training using latent_factor method with latent_factor size {}".format(rmse, latent_factor_size))
