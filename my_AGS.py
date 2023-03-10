import math
import random
import heapq
import copy
import normalize as my_norm
import nearest_neighbor as my_nn

def InitHeap(data, num_instances, num_features, current_set_of_features):
    """
    This function initializes a heap with the features not in the current set of features and their corresponding accuracies.
    """
    my_queue = []
    for i in range(1, num_features + 1):
        if (i not in current_set_of_features):
            accuracy = my_nn.one_out_cross_validation(data, num_instances, \
                current_set_of_features, i)
            # The feature pair is a tuple of negative accuracy (to use a min heap) and feature index
            feature_pair = (1 - accuracy, i)
            heapq.heappush(my_queue, feature_pair)
    return my_queue

def SampleHeap(data, num_instances, num_features, current_set_of_features, my_queue):
    """
    This function pops the feature with the lowest accuracy from the heap and returns its index and accuracy.
    """
    top = heapq.heappop(my_queue)
    feature_to_add = top[1]
    accuracy = my_nn.one_out_cross_validation(data, num_instances, \
        current_set_of_features, feature_to_add)
    return feature_to_add, accuracy

def AdaptiveGreedySearch(data, num_instances, num_features):
    """
    This function performs Adaptive Greedy Search on the given data and returns the best set of features and its accuracy.
    """
    current_set_of_features = set()
    queue_features = InitHeap(data, num_instances, num_features, \
        current_set_of_features)
    top = heapq.heappop(queue_features)
    feature_to_add = top[1]
    best_so_far_accuracy = 1 - top[0]
    current_set_of_features.add(feature_to_add)

    recently_reordered = False
    num_reorder = 0
    reorder_limit = 1
    iteration = 1
    depth = 2

    while (len(queue_features) != 0 and len(current_set_of_features) < 5):
        improved_accuracy = False
        for i in range(1, depth+1):
            feature_to_add, accuracy = SampleHeap(data, num_instances, num_features,\
                current_set_of_features, queue_features)
            if (accuracy > best_so_far_accuracy):
                current_set_of_features.add(feature_to_add)
                best_so_far_accuracy = accuracy
                improved_accuracy = True
                break
        if (recently_reordered and not improved_accuracy):
            # Stop the search if the heap has been reordered recently and no improvement has been made
            break
        if (not improved_accuracy and num_reorder < reorder_limit):
            # Reinitialize the heap with the updated current set of features if no improvement has been made and the limit of reorders has not been reached
            queue_features = InitHeap(data, num_instances, num_features,\
                current_set_of_features)
            recently_reordered = True
            num_reorder += 1
        else:
            current_set_of_features.add(feature_to_add)
            best_so_far_accuracy = accuracy
        iteration += 1
    return current_set_of_features, best_so_far_accuracy
