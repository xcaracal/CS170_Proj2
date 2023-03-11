import matplotlib.pyplot as plt

def plot_features(data):
    num_features = len(data[0]) - 1
    for i in range(1, num_features + 1):
        for j in range(i+1, num_features + 1):
            plt.figure()
            plt.title(f"Feature {i} vs Feature {j}")
            plt.xlabel(f"Feature {i}")
            plt.ylabel(f"Feature {j}")
            plt.scatter([row[i] for row in data], [row[j] for row in data], c=[row[0] for row in data])
            plt.show()
