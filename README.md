## Introduction
In today’s digital world, it is common to see a lot of recommendations suggested to us based on our previous behaviour. As this provides huge profits for the companies, all online commercial giants are fighting to get better at solving the same problem: given observation of users’ past behaviour, predict what other things the same user will like.

There are two main types of recommendation systems:
### Content-based system
This is an intuitive method, where information that we know about people are used as a connectivity for recommendations. The downside for this approach is that it’s overly simplified and not very accurate. An obvious solution is to increase the number of labelling attributes, but that requires collection of user data and keep track of all the attributes.
### Collaborative filtering system
Therefore, a second method is introduced which is called collaborative filtering system. This method doesn’t require to keep track of all the attributes as for the content-based system. As the name suggested, this method focus on collaboration where the system will recommend items based on users that are alike.

In this rep, I will perform the implementation on the two collaborative filtering system.

## How to run the program
### Install dependencies

### Run on terminal
`python3 latent_factor_method.py` or `python3 neighbourhood_model.py`

## Explanation for two methods
### Neighbourhood Method
- The program will loop through each user and find all the cosine similarities between this user and the others. This value will be stored in a N*N matrix where N is the number of users.

- The program will then pick the top K users with the highest similarities with this user and store their value in an array called top_n_neighbours. The value K will be determined by a pre-set value neighbourhood_size, and we will change this value to compare the performance of the algorithm.

- Finally, the program will calculate the predicted rating using the formula mentioned in 2.1.4 and update the original train_matrix value for this user item match according.

- The program will calculate the RMSE of the train_matrix and the test_matrix, which was separated randomly when processing the dataset.

### Latent Factor Method
- First, we create two array called permutation_u and permutation_v, which stores the combination of (i, j), where i is the row number of u/v and j is the column number of u/v. After creating these two arrays, the program will shuffle them, which is for later usage of the cell optimisation.


- To go through the iteration, we check if the difference between previous_rmse and the current rmse is larger than the threshold 0.005, if so then we continue with the iteration.

- Inside the iteration, the program go through the random location of U and V, and uses the formula introduced in 2.2.2 to optimise the RMSE.

- The RMSE is not guaranteed to be the global minima, we repeat this process 5 times with the same latent_factor_size and get the average RMSE value.