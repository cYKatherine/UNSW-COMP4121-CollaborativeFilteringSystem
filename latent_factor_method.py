import copy
import csv
import math
import numpy as np
import sys
import random
import os

def write_to_output_file(output_file_name, matrix):
    f = open(output_file_name, 'w')
    writer = csv.writer(f)
    for line in matrix:
      writer.writerow(line)
    f.close()

if __name__ == "__main__":
    train_matrix = []
    test_matrix = []
    latent_factor_size = 15

    file_name = "google_review_ratings_test_small.csv" # the original file
    train_dataset_file_name = "latent_factor_train_data_before_prediction.csv"

    f = open(file_name, 'r')
    reader = csv.reader(f)
    for line in reader:
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
    # Output the latent_factor_train_data_before_prediction.csv file
    write_to_output_file("latent_factor_train_data_after_prediction.csv", train_matrix)

    rmse = 0
    for i in range(len(test_matrix)):
        for j in range(len(test_matrix[0])):
            rmse += (train_matrix[i][j] - test_matrix[i][j]) ** 2
        rmse /= len(train_matrix)
        rmse = math.sqrt(rmse)
    print("RMSE is {} after training using latent_factor method with latent_factor size {}".format(rmse, latent_factor_size))
