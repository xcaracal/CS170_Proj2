import math

def nearest_neighbor(instances, num_instances, one_out, features):
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
    return instances[nearest_neighbor_index][0] == instances[one_out][0]

def one_out_cross_validation(instances, num_instances, current_features, my_feature):
    """
    Pass in positive to add, negative to remove, 0 for no feature
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
    print(f"Testing features: {list_features} with accuracy {accuracy:.2f}")
    return accuracy
