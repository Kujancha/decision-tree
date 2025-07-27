import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, *, value = None):
        
        #---------------------
        # ONLY FOR LEAF NODE 
        # every node should have a feature, i.e cols of the data () and threshold, i.e conditions of split
        #------------------------------
        self.feature = feature
        self.threshold = threshold
        
        # this is for the children
        self.left = left
        self.right = right
        
        # this is ONLY for the leaf node
        self.value = value
        
    def is_leaf(self):
        return self.value is not None
        
# some terminologies may be necessary to be introduced
#------------
# sample = the rows in the training data (exlcuding the label) (no of data)
# label = the part of data that contains the decision, or what we are trying to predict. (maybe its the type of flower, or the nature of a number)        
# features = the columns of the training data (the no. of factors that affect the label)

class DecisionTree:
    def __init__(self, max_depth = 10, min_sample_split = 2):
        
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split   # the tree should have at least these many these samples
        
        self.root = None # root node 
        
    # some helper functions
    
    def _entropy(self, label):
        #-----------calculates the entropy of an array named label
        hist = np.bincount(label)  # returns a list result such that result[i] tells how many quantity of i is in label
        probabilities = hist/len(label)     # prob. of occorunce of each data
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p>0])  # entropy = - sum(p(x * long2(p(x))))
        return entropy
    
    
    def _most_common_label(self, label):
        #reutnrs the most commonnn labell lol
        counter = Counter(label)
        return counter.most_common(1)[0][0]
    
    
    def _split(self, X_columns, threshold):
        # Split indices of X based on a threshold on one feature
        left_indices = np.argwhere(X_columns <= threshold).flatten()
        right_indices = np.argwhere(X_columns > threshold).flatten()
        
        return left_indices, right_indices
    
    
    def _information_gain(self, label, left_indices, right_indices):
        # calculate the info gain when label is split into left and right
        
        n = len(label)
        n_left, n_right = len(left_indices), len(right_indices)
        
        if  n_left == 0 or n_right == 0:
            return 0
        
        parent_entropy = self._entropy(label)
        entropy_left = self._entropy(label[left_indices])
        entropy_right = self._entropy(label[right_indices])
        
        child_entropy = (n_left/ n) * entropy_left + (n_right / n) * entropy_right
        
        return parent_entropy - child_entropy
    
    def _best_split(self, data, label):
        # tries all possible split and returns one with the most IG
        
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = data.shape
        
        for feature in range(n_features):
            thresholds = np.unique(data[:,feature])  # extracts all the rows of all cols
            
            for threshold in thresholds:
                left_idxs, right_idxs = self._split(data[:,feature], threshold)
                gain = self._information_gain(label, left_idxs, right_idxs)
                
                if gain>best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        
        return best_feature, best_threshold
    
    # NOW THE FUNCTIONS TO BUILD THE TREE ARE HERERE LOLL
    def _build_tree(self, data, label, depth = 0):
        n_samples, n_features = data.shape
        num_classes = len(set(label))
        
        # stopping conditind (i ahve very good spelling lol)
        if(depth > self.max_depth or n_samples < self.min_sample_split or num_classes == 1):
            leaf_value = self._most_common_label(label)
            return Node(value=leaf_value)
        
        best_feature, best_threshold = self._best_split(data, label)
        
        if best_feature is None:
            leaf_value = self._most_common_label(label)
            return Node(value=leaf_value)
        
        #recursion parttt
        left_idxs, right_idxs = self._split(data[:, best_feature], best_threshold)
        left = self._build_tree(data[left_idxs], label[left_idxs], depth=depth+1)
        right = self._build_tree(data[right_idxs], label[right_idxs], depth=depth+1)
        
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)
    
    def _traverse_tree(self, x, node:Node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <=node.threshold:
            return self._traverse_tree(x, node.left)
            
        else:
            return self._traverse_tree(x, node.right)
            
            
            
    
    # now the actual fitting parttttt
    def fit(self, data, label):
        self.root = self._build_tree(data,label)
        
    def predict(self, testData):
        results = []
        for x in testData:
            print("Predicting for:", x)
            result = self._traverse_tree(x, self.root)
            print("Result:", result)
            results.append(result)
        return np.array(results)       
            
      
      
        
        