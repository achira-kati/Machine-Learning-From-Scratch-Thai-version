import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# plt function from coursera
def draw_line(p1, p2, style="-k", linewidth=1):
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], style, linewidth=linewidth)

def plot_data_points(X, idx):
    # Define colormap to match Figure 1 in the notebook
    cmap = ListedColormap(["red", "green", "blue"])
    c = cmap(idx)
    
    # plots data points in X, coloring them so that those with the same
    # index assignments in idx have the same color
    plt.scatter(X[:, 0], X[:, 1], facecolors='none', edgecolors=c, linewidth=0.1, alpha=0.7)

def plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i):
    # Plot the examples
    plot_data_points(X, idx)
    
    # Plot the centroids as black 'x's
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='k', linewidths=3)
    
    # Plot history of the centroids with lines
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous_centroids[j, :])
    
    plt.title("Iteration number %d" %i)


def closest_centroid(X, centroid):
	m = len(X)
	idx = np.zeros(m, dtype=int)
	for i in range(m):
		dis = []
		for j in range(len(centroid)):
			distance = np.linalg.norm(X[i] - centroid[j], ord=2)
			dis.append(distance)
		idx[i] = np.argmin(dis)
	return idx

def update_centroid(X, idx, K):
	n = X.shape[1]
	centroid = np.zeros((K, n))
	for i in range(len(centroid)):
		data_in_c = X[idx == i]
		centroid[i] = np.mean(data_in_c, axis=0)
	return centroid

def init_centroid(X, K):
	m = X.shape[0]
	randidx = np.random.permutation(m)
	centroid = X[randidx[:K]]
	return centroid

class KMean:
	def __init__(self, K=3, max_iters=100):
		self.K = K
		self.max_iters = max_iters
		np.random.seed(12345)
	
	def fit(self, X):
		centeroid = init_centroid(X, self.K)
		prev = centeroid
		for i in range(self.max_iters):
			idx = closest_centroid(X, centeroid)
			plot_progress_kMeans(X, centeroid, prev, idx, self.K, i)
			prev = centeroid
			centeroid = update_centroid(X, idx, self.K)
		plt.show()