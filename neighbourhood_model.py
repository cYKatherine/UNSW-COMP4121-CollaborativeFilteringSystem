import copy
import csv
import math
import numpy as np
import sys
import random
import os
import re
from scipy.spatial import distance

def find_index_of_top_n_neighbours(neighbourhood_size, user_index, similarities):
    res = np.argsort(similarities[user_index])[-neighbourhood_size:]
    # print("The {} most similar neightbours for user {} are user {}".format(neighbourhood_size, user_index, res))
    return res

def compute(neighbourhood_size, train_matrix):
    no_of_users = len(train_matrix)
    similarities = []
    for i in range(no_of_users):
        similarities.append([])
        for j in range(no_of_users):
            similarities[i].append(distance.cosine(train_matrix[i], train_matrix[j]))
    
    for user_index in range(no_of_users):
        top_n_neighbours = find_index_of_top_n_neighbours(neighbourhood_size, user_index, similarities)
        for i in range(len(train_matrix[user_index])):
            user_rating = train_matrix[user_index][i]
            if user_rating == 0: # This rating is unknown
                predicted_rating = 0
                temp = 0
                for similar_user_index in top_n_neighbours:
                    predicted_rating += similarities[user_index][similar_user_index] * train_matrix[similar_user_index][i]
                    temp += similarities[user_index][similar_user_index]
                predicted_rating /= temp
                train_matrix[user_index][i] = predicted_rating

def write_to_output_file(output_file_name, matrix):
    f = open(output_file_name, 'w')
    writer = csv.writer(f)
    for line in matrix:
      writer.writerow(line)
    f.close()

if __name__ == "__main__":
    train_matrix = []
    test_matrix = []
    neighbourhood_size = 10

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

    compute(neighbourhood_size, train_matrix)
    # Output the train_data_before_prediction.csv file
    write_to_output_file("neighbourhood_train_data_after_prediction.csv", train_matrix)

    rmse = 0
    for i in range(len(test_matrix)):
        for j in range(len(test_matrix[0])):
            rmse += (train_matrix[i][j] - test_matrix[i][j]) ** 2
        rmse /= len(train_matrix)
        rmse = round(math.sqrt(rmse),2)
    print("RMSE is {} after training using neighbourhood method with neighbourhood size {}".format(rmse, neighbourhood_size))