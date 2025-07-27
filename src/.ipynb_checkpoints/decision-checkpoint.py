from graphviz import Digraph
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        return self.value is not None


class GraphedDecisionTree:
    def __init__(self, max_depth=10, min_sample_split=2):
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.root = None

    def _entropy(self, label):
        hist = np.bincount(label)
        probabilities = hist / len(label)
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        return entropy

    def _most_common_label(self, label):
        counter = Counter(label)
        return counter.most_common(1)[0][0]

    def _split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def _information_gain(self, label, left_idxs, right_idxs):
        n = len(label)
        n_left, n_right = len(left_idxs), len(right_idxs)
        if n_left == 0 or n_right == 0:
            return 0
        parent_entropy = self._entropy(label)
        entropy_left = self._entropy(label[left_idxs])
        entropy_right = self._entropy(label[right_idxs])
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right
        return parent_entropy - child_entropy

    def _best_split(self, data, label):
        best_gain = -1
        best_feature = None
        best_threshold = None
        n_samples, n_features = data.shape
        for feature in range(n_features):
            thresholds = np.unique(data[:, feature])
            for threshold in thresholds:
                left_idxs, right_idxs = self._split(data[:, feature], threshold)
                gain = self._information_gain(label, left_idxs, right_idxs)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _build_tree(self, data, label, depth=0):
        n_samples, n_features = data.shape
        num_classes = len(set(label))
        if (depth >= self.max_depth or
            n_samples < self.min_sample_split or
            num_classes == 1):
            leaf_value = self._most_common_label(label)
            return Node(value=leaf_value)
        best_feature, best_threshold = self._best_split(data, label)
        if best_feature is None:
            leaf_value = self._most_common_label(label)
            return Node(value=leaf_value)
        left_idxs, right_idxs = self._split(data[:, best_feature], best_threshold)
        left = self._build_tree(data[left_idxs], label[left_idxs], depth + 1)
        right = self._build_tree(data[right_idxs], label[right_idxs], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _traverse_tree(self, x, node: Node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    def fit(self, data, label):
        self.root = self._build_tree(data, label)

    def predict(self, testData):
        return np.array([self._traverse_tree(x, self.root) for x in testData])

    def export_graphviz(self, out_file="tree", feature_names=None):
        dot = Digraph()

        def add_nodes_edges(node, counter):
            if node.is_leaf():
                node_id = str(counter[0])
                dot.node(node_id, label=f"Leaf\nValue: {node.value}", shape="box", style="filled", color="lightblue")
                return node_id
            else:
                node_id = str(counter[0])
                feature_name = feature_names[node.feature] if feature_names else f"X[{node.feature}]"
                label = f"{feature_name} <= {node.threshold}"
                dot.node(node_id, label=label)
                counter[0] += 1
                left_id = add_nodes_edges(node.left, counter)
                dot.edge(node_id, left_id, label="True")
                counter[0] += 1
                right_id = add_nodes_edges(node.right, counter)
                dot.edge(node_id, right_id, label="False")
                return node_id

        add_nodes_edges(self.root, counter=[0])
        dot.render(out_file, format="png", cleanup=True)
        print(f"Tree saved as {out_file}.png")
