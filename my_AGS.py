import math
import random
import heapq
import copy
import normalize as my_norm
import nearest_neighbor as my_nn

def InitHeap(data, number_of_instances, number_of_features, curr_set_features):
    """
    This function initializes a heap with the features not in the current set of features and their corresponding accuracies.
    """
    _queue = []
    for i in range(1, number_of_features + 1):
        if (i not in curr_set_features):
            accuracy = my_nn.one_out_cross_validation(data, number_of_instances, \
                curr_set_features, i)
            # The feature pair is a tuple of negative accuracy (to use a min heap) and feature index
            feature_pair = (1 - accuracy, i)
            heapq.heappush(_queue, feature_pair)
    return _queue

def SampleHeap(data, number_of_instances, number_of_features, curr_set_features, _queue):
    """
    This function pops the feature with the lowest accuracy from the heap and returns its index and accuracy.
    """
    top = heapq.heappop(_queue)
    feature_add = top[1]
    accuracy = my_nn.one_out_cross_validation(data, number_of_instances, \
        curr_set_features, feature_add)
    return feature_add, accuracy

def AdaptiveGreedySearch(data, number_of_instances, number_of_features):
    """
    This function performs Adaptive Greedy Search on the given data and returns the best set of features and its accuracy.
    """
    curr_set_features = set()
    queue_features = InitHeap(data, number_of_instances, number_of_features, \
        curr_set_features)
    top = heapq.heappop(queue_features)
    feature_to_add = top[1]
    best_accuracy = 1 - top[0]
    curr_set_features.add(feature_to_add)

    recently_reordered = False
    number_reorder = 0
    reorder_limit = 1
    depth = 2
    iteration = 1

    while (len(queue_features) != 0 and len(curr_set_features) < 5):
        improved_accuracy = False
        for i in range(1, depth+1):
            feature_to_add, accuracy = SampleHeap(data, number_of_instances, number_of_features,\
                curr_set_features, queue_features)
            if (accuracy > best_accuracy):
                curr_set_features.add(feature_to_add)
                best_accuracy = accuracy
                improved_accuracy = True
                break
        if (recently_reordered and not improved_accuracy):
            # Stop the search if the heap has been reordered recently and no improvement has been made
            break
        if (not improved_accuracy and number_reorder < reorder_limit):
            # Reinitialize the heap with the updated current set of features if no improvement has been made and the limit of reorders has not been reached
            queue_features = InitHeap(data, number_of_instances, number_of_features,\
                curr_set_features)
            recently_reordered = True
            number_reorder += 1
        else:
            curr_set_features.add(feature_to_add)
            best_accuracy = accuracy
        iteration += 1
    return curr_set_features, best_accuracy
