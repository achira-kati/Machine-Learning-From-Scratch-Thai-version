import numpy as np


def entropy(y):
	entropy = 0
	if len(y) != 0:
		p1 = len(y[y==1]) / len(y)
		if p1 != 0 and p1 != 1:
			entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
	return entropy

def split_data(X, node_indice, feature):
	left = []
	right = []
	
	for i in node_indice:
		if X[i, feature] == 1:
			left.append(i)
		else:
			right.append(i)
	return left, right


def information_gain(X, y, node_indice, feature):
	#Split data
	left_indice, right_indice = split_data(X, node_indice, feature)
	X_node, y_node = X[node_indice], y[node_indice]
	X_left, y_left = X[left_indice], y[left_indice]
	X_right, y_right = X[right_indice], y[right_indice]
	entropy_node = entropy(y_node)
	w_left = len(X_left) / len(X_node)
	entropy_left =  entropy(y_left)
	w_right = len(X_right) / len(X_node)
	entropy_right = entropy(y_right)
	return entropy_node - ((w_left * entropy_left) + (w_right * entropy_right))

def get_best_feature(X, y, node_indice):
	feature = X.shape[1]
	max_info = 0
	best_feature = 0
	for i in range(feature):
		info_gain = information_gain(X, y, node_indice, i)
		if info_gain > max_info:
			max_info = info_gain
			best_feature = i
	return best_feature

class DecisionTree:
	
	def fit(self, X, y, node_indice, max_depth, curr_depth, name):
		if curr_depth == max_depth:
			print(f'{name} leaf node is {node_indice}')
			return
		# get best feature
		best_feature = get_best_feature(X, y, node_indice)
		print(f'Split {name} on feature {best_feature}')
		# split data via best feature
		left, right = split_data(X, node_indice, best_feature)

		# call fit function until reach max depth
		self.fit(X, y, left, max_depth, curr_depth + 1, 'Left')
		self.fit(X, y, right, max_depth, curr_depth + 1, 'Right')

