import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.split_counts = {}

    def fit(self, X, y):
        self.split_counts = {i: 0 for i in range(X.shape[1])}
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        if depth >= self.max_depth or n_labels <= 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))

        feat_idxs = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        if best_feat is None:
            return Node(value=self._most_common_label(y))

        self.split_counts[best_feat] += 1
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (len(left_idxs) / n) * e_l + (len(right_idxs) / n) * e_r
        return parent_entropy - child_entropy

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        if len(y) == 0:
            return 0
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class CustomRandomForest:
    def __init__(self, n_trees=5, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.classes_ = []
        self.feature_importances_ = None

    def fit(self, X, y):
        self.trees = []
        self.classes_ = np.unique(y)
        self.feature_importances_ = np.zeros(X.shape[1])
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth)
            X_samp, y_samp = self._bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)
            for feat, count in tree.split_counts.items():
                self.feature_importances_[feat] += count
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ = self.feature_importances_ / np.sum(self.feature_importances_)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([np.bincount(pred).argmax() for pred in tree_preds])

    def predict_proba(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        probas = []
        for sample_preds in tree_preds:
            counts = {c: 0 for c in self.classes_}
            for pred in sample_preds:
                counts[pred] += 1
            total = sum(counts.values())
            probas.append([counts[c] / total for c in self.classes_])
        return np.array(probas)
