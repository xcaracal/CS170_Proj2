import math
from typing import List


def calc_mean(instances: List[List[float]], num_instances: int, num_features: int) -> List[float]:
    means = []
    for i in range(1, num_features + 1):
        feature_sum = sum(row[i] for row in instances)
        means.append(feature_sum / num_instances)
    return means


def calc_std(instances: List[List[float]], num_instances: int, num_features: int, means: List[float]) -> List[float]:
    stds = []
    for i in range(1, num_features + 1):
        variance_sum = sum(pow((row[i] - means[i-1]), 2) for row in instances)
        stds.append(math.sqrt(variance_sum / num_instances))
    return stds


def normalize_data(instances: List[List[float]], num_instances: int, num_features: int) -> List[List[float]]:
    normalized_instances = [list(instance) for instance in instances]
    means = calc_mean(instances, num_instances, num_features)
    stds = calc_std(instances, num_instances, num_features, means)
    for i in range(num_instances):
        for j in range(1, num_features + 1):
            normalized_instances[i][j] = ((instances[i][j] - means[j-1]) / stds[j-1])
    return normalized_instances
