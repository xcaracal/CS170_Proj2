import math
from typing import List


def calc_mean(instances: List[List[float]], number_of_instances: int, number_of_features: int) -> List[float]:
    """
    This function calculates the means of the given instances for each feature.
    """
    mean = []
    for i in range(1, number_of_features + 1):
        feature_sum = sum(row[i] for row in instances)
        mean.append(feature_sum / number_of_instances)
    return mean


def calc_std(instances: List[List[float]], number_of_instances: int, number_of_features: int, mean: List[float]) -> List[float]:
    """
    This function calculates the standard deviations of the given instances for each feature using the given means.
    """
    stds = []
    for i in range(1, number_of_features + 1):
        variance = sum(pow((row[i] - mean[i-1]), 2) for row in instances)
        stds.append(math.sqrt(variance / number_of_instances))
    return stds


def normalize_data(instances: List[List[float]], number_of_instances: int, number_of_features: int) -> List[List[float]]:
    """
    This function normalizes the given instances using mean normalization (subtracting the mean and dividing by the standard deviation).
    """
    normalized_instance = [list(instance) for instance in instances]
    means = calc_mean(instances, number_of_instances, number_of_features)
    stds = calc_std(instances, number_of_instances, number_of_features, means)
    for i in range(number_of_instances):
        for j in range(1, number_of_features + 1):
            normalized_instance[i][j] = ((instances[i][j] - means[j-1]) / stds[j-1])
    return normalized_instance
