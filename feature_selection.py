# Import necessary libraries and modules
from pprint import pprint
import math
import random
import normalize as my_norm
import nearest_neighbor as my_nn
import my_AGS
from my_data import load_data
from plot import plot_features

def load_data(file_name, num_instances):
    """
    This function reads the data from the given file and returns it as a list of lists.
    """
    try:
        with open(file_name, 'r') as f:
            # Read the data from the file line by line and convert each value to a float
            instances = [[float(j) for j in f.readline().split()] for _ in range(num_instances)]
    except FileNotFoundError:
        # Raise an error if the file is not found
        raise FileNotFoundError(file_name)
    return instances

def forward_selection(data, number_of_instances, number_of_features):
    """
    This function performs forward feature selection on the given data and prints the best set of features and its accuracy.
    """
    print("---------------------------------------------------------")
    curr_set_features = set()
    best_accuracy = 0
    print("---------------------------------------------------------")

    for i in range(number_of_features):
        print(f"level: {i+1} of search tree. Our set is {curr_set_features}")
        add_feature = -1

        # Iterate through each feature and check if adding it to the current set of features gives better accuracy
        for j in range(1, number_of_features + 1):
            if j not in curr_set_features:
                accuracy = my_nn.one_out_cross_validation(data, number_of_instances, curr_set_features, j)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    add_feature = j

        if add_feature > 0:
            curr_set_features.add(add_feature)
            print(f"level: {i+1} of search tree, add feature {add_feature} accuracy is now: {best_accuracy}")
            print("-" * 50)
        else:
            print("NOTE:Stop - Accurracy is decreasing!!! :(")
            break

    print("---------------------------------------------------------")
    print(f"Best set of features: {curr_set_features} accuracy is now: {best_accuracy}")

def backward_elimination(data, number_of_instances, number_of_features):
    """
    This function performs backward feature elimination on the given data and prints the best set of features and its accuracy.
    """
    print("---------------------------------------------------------")
    curr_set_features = set(i+1 for i in range(0, number_of_features))
    best_accuracy = 0
    print("-" * 50)

    for i in range(number_of_features):
        print(f"level {i+1} of search tree. Our set is {curr_set_features}")
        feature_remove = -1

        # Iterate through each feature and check if removing it from the current set of features gives better accuracy
        for j in range(1, number_of_features + 1):
            if j in curr_set_features:
                accuracy = my_nn.one_out_cross_validation(data, number_of_instances, curr_set_features, (-1 * j))

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    feature_remove = j

        if feature_remove > 0:
            curr_set_features.remove(feature_remove)
            print(f"level {i+1} of search tree. remove feature {feature_remove} accuracy is now: {best_accuracy}")
            print("-" * 50)
        else:
            print("NOTE:Stop - Accurracy is decreasing!!! :(")
            break

    print("---------------------------------------------------------")
    print(f"Best set of features are: {curr_set_features} accuracy is now: {best_accuracy}")

def main():
    data = load_data("my_dataset.txt", 100)
    plot_features(data)
    """
    This function is the main entry point of the program and prompts the user to enter the file name and algorithm to use.
    """
    file = input("Enter Test File Name: ")
    num_instances = int(input("How many instances to read?: "))
    instances = load_data(file, num_instances)

    alg = ""
    while (alg != "FS" and alg != "BE" and alg != "C"):
        # Prompt the user to choose the algorithm to use
        alg = input("""Which Algorithmn do you want to use?:
                       FS : Forward Select
                       BE : Backward Elim
                       C : Custom
                    \r""")
    num_features = len(instances[0]) - 1
    print("\t****Normalizing****")
    # Normalize the data using the normalize_data function from the normalize module
    normalized = my_norm.normalize_data(instances, num_instances, num_features)
    print("There are %d features and %d instances." % (num_features, num_instances))
    
    if (alg == "FS"):
        # Call the forward_selection function if the user chooses forward selection
        forward_selection(normalized, num_instances, num_features)
    elif (alg == "BE"):
        # Call the backward_elimination function if the user chooses backward elimination
        backward_elimination(normalized, num_instances, num_features)
    else:
        # Call the AdaptiveGreedySearch function from the my_AGS module if the user chooses custom search
        my_AGS.AdaptiveGreedySearch(normalized, num_instances, num_features)

if __name__ == '__main__':
    # Call the main function when the program is run
    main()
