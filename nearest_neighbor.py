import math

def nearest_neighbor(instances, num_instances, one_out, features):
    """
    This function finds the nearest neighbor of the given instance with the given features using Euclidean distance.
    """
    nearest_neighbor_i = -1
    nearest_neighbor_dist = float('inf')
    num_features = len(features)
    
    for i in range(num_instances):
        if i == one_out:
            continue
        
        distance = math.sqrt(sum((instances[i][features[j]] - instances[one_out][features[j]]) ** 2 for j in range(num_features)))
        
        if distance < nearest_neighbor_dist:
            nearest_neighbor_dist = distance
            nearest_neighbor_i = i
    
    return nearest_neighbor_i

def check_classification(instances, nearest_neighbor_i, oneout):
    """
    This function checks if the classification of the nearest neighbor of the given instance is the same as the instance's classification.
    """
    return instances[nearest_neighbor_i][0] == instances[oneout][0]

def one_out_cross_validation(instances, number_of_instances, curr_features, the_feature):
    """
    This function performs one-out cross validation using the given features with the given feature added, removed, or not changed.
    """
    if the_feature > 0:
        list_features = list(curr_features) + [the_feature]
    elif the_feature < 0:
        list_features = list(curr_features)[:]
        list_features.remove(-the_feature)
    else:
        list_features = list(curr_features)
    
    num_correct = 0
    
    for i in range(number_of_instances):
        one_out = i
        nearest_neighbor_index = nearest_neighbor(instances, number_of_instances, one_out, list_features)
        correct_classification = check_classification(instances, nearest_neighbor_index, one_out)
        num_correct += correct_classification
    
    accuracy = num_correct / number_of_instances
    print(f"Testing these features: {list_features} accuracy is: {accuracy:.2f}")
    return accuracy
