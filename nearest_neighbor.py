import math

def nearest_neighbor(instances, num_instances, one_out, features):
    """
    This function finds the nearest neighbor of the given instance with the given features using Euclidean distance.
    """
    nearest_neighbor_index = -1
    nearest_neighbor_distance = float('inf')
    num_features = len(features)
    
    for i in range(num_instances):
        if i == one_out:
            continue
        
        distance = math.sqrt(sum((instances[i][features[j]] - instances[one_out][features[j]]) ** 2 for j in range(num_features)))
        
        if distance < nearest_neighbor_distance:
            nearest_neighbor_distance = distance
            nearest_neighbor_index = i
    
    return nearest_neighbor_index

def check_classification(instances, nearest_neighbor_index, one_out):
    """
    This function checks if the classification of the nearest neighbor of the given instance is the same as the instance's classification.
    """
    return instances[nearest_neighbor_index][0] == instances[one_out][0]

def one_out_cross_validation(instances, num_instances, current_features, my_feature):
    """
    This function performs one-out cross validation using the given features with the given feature added, removed, or not changed.
    """
    if my_feature > 0:
        list_features = list(current_features) + [my_feature]
    elif my_feature < 0:
        list_features = list(current_features)[:]
        list_features.remove(-my_feature)
    else:
        list_features = list(current_features)
    
    num_correct = 0
    
    for i in range(num_instances):
        one_out = i
        nearest_neighbor_index = nearest_neighbor(instances, num_instances, one_out, list_features)
        correct_classification = check_classification(instances, nearest_neighbor_index, one_out)
        num_correct += correct_classification
    
    accuracy = num_correct / num_instances
    print(f"Testing these features: {list_features} accuracy is: {accuracy:.2f}")
    return accuracy
