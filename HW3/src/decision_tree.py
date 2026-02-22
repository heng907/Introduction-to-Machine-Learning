import numpy as np
import matplotlib.pyplot as plt

class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_labels = len(np.unique(y)) # the number of different labels in y
        # === stop grow condition ===
        if n_labels == 1 or n_samples == 0:
            leaf_val = majority_vote(y)
            return {"leaf": True, "value": leaf_val}
        # reach the max depth
        if depth >= self.max_depth:
            leaf_val = majority_vote(y)
            return {"leaf": True, "value": leaf_val}
        
        # === find the best split ===
        best_feature, best_threshold, best_gain = find_best_split(X, y)

        if best_feature is None or best_gain <= 0:
            leaf_val = majority_vote(y)
            return {"leaf": True, "value": leaf_val}
        # === split data ===
        # using best feature and threshold to split tree
        X_left, y_left, X_right, y_right = split_dataset(
            X, y, best_feature, best_threshold
        )

        # recursive build tree
        left_subtree = self._grow_tree(X_left, y_left, depth+1)
        right_subtree = self._grow_tree(X_right, y_right, depth+1)
        
        return {
            "leaf": False,
            "feature_index": best_feature,
            "threshold": best_threshold,
            "left": left_subtree,
            "right": right_subtree,
        }
    
    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree_node):
        if tree_node["leaf"]:
            return tree_node["value"]

        feature_index = tree_node["feature_index"]
        threshold = tree_node["threshold"]

        if x[feature_index] <= threshold:
            return self._predict_tree(x, tree_node["left"])
        else:
            return self._predict_tree(x, tree_node["right"])  


# Split dataset based on a feature and threshold
def split_dataset(X, y, feature_index, threshold):
    """
    take a feature_index, and then 
    devide to left and right by threshold
    """
    feature_val = X[:, feature_index]
    left_mask = feature_val <= threshold
    right_mask = feature_val > threshold

    # left part
    X_left = X[left_mask]
    y_left = y[left_mask]
    # right part
    X_right = X[right_mask]
    y_right = y[right_mask]

    return X_left, y_left, X_right, y_right

# Find the best split for the dataset
def find_best_split(X, y):
    """
    find the biggest information gain
    and return the best combination of 
    (feature_index, threshold, gain)
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]

    if n_samples <= 1:
        return None, None, 0.0
    
    parent_gini = gini(y)
    best_gain = 0.0
    best_feature = None
    best_threshold = None

    for feature_index in range(n_features):
        feature_val = X[:, feature_index]
        thresholds = np.unique(feature_val)

        for threshold in thresholds:
            X_left, y_left, X_right, y_right = split_dataset(
                X, y, feature_index, threshold
            )

            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            # calculate information gain
            gain = gini_gain(y, y_left, y_right, parent_gini)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold, best_gain

def gini(y):
    """
    Gini impurity:
    G(y) = 1 - sum_k p_k^2
    """
    if len(y) == 0:
        return 0.0

    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1.0 - np.sum(probs ** 2)


def gini_gain(y, y_left, y_right, parent_gini=None):
    """
    Gini-based "gain":
    ΔG = G(parent) - (n_left/n)*G(left) - (n_right/n)*G(right)
    """
    n = len(y)
    if n == 0:
        return 0.0

    if parent_gini is None:
        parent_gini = gini(y)

    n_left = len(y_left)
    n_right = len(y_right)

    if n_left == 0 or n_right == 0:
        return 0.0

    G_left = gini(y_left)
    G_right = gini(y_right)

    child_gini = (n_left / n) * G_left + (n_right / n) * G_right

    return parent_gini - child_gini


def entropy(y):
    """
    H(y) = - sum_k p_k log2 p_k
    """
    eps = 1e-10 # avoid log(0)
    if len(y) == 0:
        return 0.0
    val, cnt = np.unique(y, return_counts=True)
    prob = cnt / cnt.sum()

    return -np.sum(prob * np.log2(prob+eps))

def majority_vote(y):
    """
    return the most counts 
    of class(0 or 1) in y
    """
    if len(y) == 0:
        return 0
    values, cnt = np.unique(y, return_counts=True)
    return values[np.argmax(cnt)]

def compute_tree_feature_importance(tree, num_features):
    importance = np.zeros(num_features, dtype=np.float32)

    def traverse(node):
        if node["leaf"]:
            return
        idx = node["feature_index"]
        importance[idx] += 1
        traverse(node["left"])
        traverse(node["right"])

    traverse(tree)

    if importance.sum() > 0:
        importance = importance / importance.sum()

    return importance


def aggregate_feature_importance(onehot_importance, feature_names):

    original_features = [
        "person_age", "person_gender", "person_education", "person_income",
        "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
        "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
        "credit_score", "previous_loan_defaults_on_file"
    ]

    agg_scores = []

    for orig_feat in original_features:
        score = 0.0
        for idx, feat in enumerate(feature_names):
            if feat.startswith(orig_feat):
                score += float(onehot_importance[idx])
        agg_scores.append(score)

    return original_features, agg_scores


def plot_dt_feature_importance(
        tree, 
        feature_names, 
        save_path="./dt_feature_importance.png"
):
    onehot_importance = compute_tree_feature_importance(tree, len(feature_names))
    labels, agg_scores = aggregate_feature_importance(onehot_importance, feature_names)

    plt.figure(figsize=(12, 7))
    plt.barh(labels, agg_scores, color="skyblue")
    plt.xlabel("Feature Importance")
    plt.title("Decision Tree Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()