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
        # update value in u and v
        # temp_u_sum = [0] * latent_factor_size
        # for k in range(latent_factor_size):
        #     for i in range(len(u)):
        #         temp_u_sum[k] += u[i][k] ** 2

        # temp_v_sum = [0] * latent_factor_size
        # for k in range(latent_factor_size):
        #     for j in range(len(v[0])):
        #         temp_v_sum[k] += v[k][j] ** 2

        # new_u = [[0] * latent_factor_size for _ in range(len(train_matrix))] # u: i*k
        # for i in range(len(train_matrix)):
        #     for k in range(latent_factor_size):
        #         temp = 0
        #         for j in range(len(train_matrix[0])):
        #             temp2 = 0
        #             for xxx in range(k):
        #                 if xxx != k:
        #                     temp2 += u[i][xxx] * v[xxx][j]
        #             if train_matrix[i][j] != 0:
        #                 temp += v[k][j] * (train_matrix[i][j] - temp2)
        #         new_u[i][k] = temp / temp_v_sum[k]
        #         if new_u[i][k] < 0:
        #             new_u[i][k] = 0
        
        # new_v = [[0] * len(train_matrix[0]) for _ in range(latent_factor_size)] # v: k*j
        # for k in range(latent_factor_size):
        #     for j in range(len(train_matrix[0])):
        #         temp = 0
        #         for i in range(len(train_matrix)):
        #             temp2 = 0
        #             for xxx in range(k):
        #                 if xxx != k:
        #                     temp2 += u[i][xxx] * v[xxx][j]
        #             if train_matrix[i][j] != 0:
        #                 temp += u[i][k] * (train_matrix[i][j] - temp2)
        #         new_v[k][j] = temp / temp_u_sum[k]
        #         if new_v[k][j] < 0:
        #             new_v[k][j] = 0

        # print("------u------")
        # for x in u:
        #     print(x)
        # print("------new u------")
        # for x in new_u:
        #     print(x)
        # print("------v------")
        # for x in v:
        #     print(x)
        # print("------new v------")
        # for x in new_v:
        #     print(x)
        new_u = [[0] * latent_factor_size for _ in range(len(train_matrix))] # u: i*k
        for row in range(len(train_matrix)):
            for col in range(latent_factor_size):
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
                        
        new_v = [[0] * len(train_matrix[0]) for _ in range(latent_factor_size)] # v: k*j
        for row in range(latent_factor_size):
            for col in range(len(train_matrix[0])):
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

        temp_train_matrix = []
        for i in range(len(train_matrix)):
            temp_train_matrix.append([])
            for j in range(len(train_matrix[0])):
                temp_value = 0
                for k in range(latent_factor_size):
                    temp_value += u[i][k] * v[k][j]
                temp_train_matrix[i].append(temp_value)

        # u = copy.deepcopy(new_u)
        # v = copy.deepcopy(new_v)

        previous_rmse = rmse
        rmse = calculate_rmse(temp_train_matrix, train_matrix)
        print(previous_rmse, rmse, previous_rmse - rmse)
    

def write_to_output_file(output_file_name, matrix):
    f = open(output_file_name, 'w')
    writer = csv.writer(f)
    for line in matrix:
      writer.writerow(line)
    f.close()

if __name__ == "__main__":
    train_matrix = []
    test_matrix = []
    latent_factor_size = 2

    file_name = "google_review_ratings_test_small.csv" # the original file
    train_dataset_file_name = "latent_factor_train_data_before_prediction.csv"

    f = open(file_name, 'r')
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    for line in reader:
        line = [re.sub('\s+', '', s) for s in line]
        test_matrix.append([eval(i) for i in line[1:]])
        train_matrix.append([eval(i) for i in line[1:]])
    f.close()

    # Output the latent_factor_test_data.csv file
    write_to_output_file("latent_factor_test_data.csv", test_matrix)

    if os.path.exists(train_dataset_file_name):
        print("latent_factor_train_data_before_prediction.csv file exists, use this dataset to train.")
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
        # Output the latent_factor_train_data_before_prediction.csv file
        write_to_output_file("latent_factor_train_data_before_prediction.csv", train_matrix)

    compute(latent_factor_size, train_matrix)
    # # Output the latent_factor_train_data_before_prediction.csv file
    # write_to_output_file("latent_factor_train_data_after_prediction.csv", train_matrix)

    # rmse = 0
    # for i in range(len(test_matrix)):
    #     for j in range(len(test_matrix[0])):
    #         rmse += (train_matrix[i][j] - test_matrix[i][j]) ** 2
    #     rmse /= len(train_matrix)
    #     rmse = math.sqrt(rmse)
    # print("RMSE is {} after training using latent_factor method with latent_factor size {}".format(rmse, latent_factor_size))
