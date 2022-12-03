import copy
import csv
import math
import numpy as np
import sys
from scipy.spatial import distance

def find_index_of_top_n_neighbours(neighbourhood_size, user_index, similarities):
    res = np.argsort(similarities[user_index])[-neighbourhood_size:]
    print("The {} most similar neightbours for user {} are user {}".format(neighbourhood_size, user_index, res))
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

if __name__ == "__main__":
    train_matrix = []
    test_matrix = []
    neighbourhood_size = 10

    test_file = "google_review_ratings_test_small.csv" # the original file
    train_file = "google_review_ratings_train_small.csv" # the file that has missing data

    f = open(test_file, 'r')
    reader = csv.reader(f)
    #next(reader, None)  # skip the headers
    for line in reader:
        test_matrix.append([eval(i) for i in line[1:]])

    f = open(train_file, 'r')
    reader = csv.reader(f)
    for line in reader:
        train_matrix.append([eval(i) for i in line[1:]])
    
    f.close()
    compute(neighbourhood_size, train_matrix)

    rmse = 0
    for i in range(len(test_matrix)):
        for j in range(len(test_matrix[0])):
            rmse += (train_matrix[i][j] - test_matrix[i][j]) * (train_matrix[i][j] - test_matrix[i][j])
        rmse /= len(train_matrix)
        rmse = math.sqrt(rmse)
    print("RMSE is {} after training using neighbourhood method with neighbourhood size {}".format(rmse, neighbourhood_size))