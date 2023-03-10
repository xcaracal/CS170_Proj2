# Import necessary libraries and modules
from pprint import pprint
import math
import random
import normalize as my_norm
import nearest_neighbor as my_nn
import my_AGS

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

def forward_selection(data, num_instances, num_features):
    """
    This function performs forward feature selection on the given data and prints the best set of features and its accuracy.
    """
    print("---------------------------------------------------------")
    current_set_of_features = set()
    best_so_far_accuracy = 0
    print("---------------------------------------------------------")

    for i in range(num_features):
        print(f"level: {i+1} of search tree. Our set is {current_set_of_features}")
        feature_to_add = -1

        # Iterate through each feature and check if adding it to the current set of features gives better accuracy
        for j in range(1, num_features + 1):
            if j not in current_set_of_features:
                accuracy = my_nn.one_out_cross_validation(data, num_instances, current_set_of_features, j)

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add = j

        if feature_to_add > 0:
            current_set_of_features.add(feature_to_add)
            print(f"level: {i+1} of search tree, add feature {feature_to_add} accuracy is now: {best_so_far_accuracy}")
            print("-" * 50)
        else:
            print("NOTE:Stop - Accurracy is decreasing!!! :(")
            break

    print("---------------------------------------------------------")
    print(f"Best set of features: {current_set_of_features} accuracy is now: {best_so_far_accuracy}")

def backward_elimination(data, num_instances, num_features):
    """
    This function performs backward feature elimination on the given data and prints the best set of features and its accuracy.
    """
    print("---------------------------------------------------------")
    current_set_of_features = set(i+1 for i in range(0, num_features))
    best_so_far_accuracy = 0
    print("-" * 50)

    for i in range(num_features):
        print(f"level {i+1} of search tree. Our set is {current_set_of_features}")
        feature_to_remove = -1

        # Iterate through each feature and check if removing it from the current set of features gives better accuracy
        for j in range(1, num_features + 1):
            if j in current_set_of_features:
                accuracy = my_nn.one_out_cross_validation(data, num_instances, current_set_of_features, (-1 * j))

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_remove = j

        if feature_to_remove > 0:
            current_set_of_features.remove(feature_to_remove)
            print(f"level {i+1} of search tree. remove feature {feature_to_remove} accuracy is now: {best_so_far_accuracy}")
            print("-" * 50)
        else:
            print("NOTE:Stop - Accurracy is decreasing!!! :(")
            break

    print("---------------------------------------------------------")
    print(f"Best set of features are: {current_set_of_features} accuracy is now: {best_so_far_accuracy}")

def main():
    """
    This function is the main entry point of the program and prompts the user to enter the file name and algorithm to use.
    """
    file_name = input("Enter Test File Name: ")
    num_instances = int(input("How many instances to read?: "))
    instances = load_data(file_name, num_instances)

    alg = ""
    while (alg != "FS" and alg != "BE" and alg != "CS"):
        # Prompt the user to choose the algorithm to use
        alg = input("""Which Algorithmn do you want to use?:
                       FS : Forward Selection
                       BE : Backward Elimination
                       CS : Custom Search
                    \r""")
    num_features = len(instances[0]) - 1
    print("\t****Normalizing****")
    # Normalize the data using the normalize_data function from the normalize module
    normalized_instances = my_norm.normalize_data(instances, num_instances, num_features)
    print("There are %d features and %d instances." % (num_features, num_instances))
    
    if (alg == "FS"):
        # Call the forward_selection function if the user chooses forward selection
        forward_selection(normalized_instances, num_instances, num_features)
    elif (alg == "BE"):
        # Call the backward_elimination function if the user chooses backward elimination
        backward_elimination(normalized_instances, num_instances, num_features)
    else:
        # Call the AdaptiveGreedySearch function from the my_AGS module if the user chooses custom search
        my_AGS.AdaptiveGreedySearch(normalized_instances, num_instances, num_features)

if __name__ == '__main__':
    # Call the main function when the program is run
    main()
