import numpy as np
import pandas as pd
from loguru import logger
import random

import torch
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import plot_learners_roc, plot_feature_importance
from src.decision_tree import gini, entropy, plot_dt_feature_importance


def main():
    """ You can control the seed for reproducibility """
    random.seed(777)
    torch.manual_seed(777)

    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target'].to_numpy()

    # feature_names = list(train_df.drop(['target'], axis=1).columns)

    """
    TODO: Implement you preprocessing function.
    """

    # print(X_train.select_dtypes(include=['object']).columns)
    cat_cols = [
        'person_gender',
        'person_education',
        'person_home_ownership',
        'loan_intent',
        'previous_loan_defaults_on_file',
    ]
    X_train_enc = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test_enc = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

    X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    # features
    X_train_np = X_train_enc.to_numpy(dtype=np.float32)
    X_test_np = X_test_enc.to_numpy(dtype=np.float32)
    # normalized
    mean = X_train_np.mean(axis=0, keepdims=True)
    std = X_train_np.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train_np - mean) / std
    X_test = (X_test_np - mean) / std

    """
    TODO: Implement your ensemble methods.
    1. You can modify the hyperparameters as you need.
    2. You must print out logs (e.g., accuracy) with loguru.
    """
    # AdaBoost
    input_dim = X_train.shape[1]
    clf_adaboost = AdaBoostClassifier(
        input_dim=input_dim,
        num_learners=10,
    )
    _ = clf_adaboost.fit(X_train, y_train)

    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    accuracy_ = (y_pred_classes == y_test).mean()
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath='./adaboost_auc.png',
    )

    feature_importance = clf_adaboost.compute_feature_importance()
    # logger.info(f'AdaBoost - Feature importance: {feature_importance}')

    plot_feature_importance(
        feature_importance=feature_importance,
        feature_names=X_train_enc.columns,
        title="AdaBoost Feature Importance",
        save_path="./adaboost_importance.png",
    )

    # Bagging
    clf_bagging = BaggingClassifier(input_dim=input_dim)
    _ = clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=700,
        learning_rate=0.002,
    )

    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = (y_pred_classes == y_test).mean()
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath='./bagging_auc.png',
    )
    feature_importance = clf_bagging.compute_feature_importance()
    # logger.info(f'Bagging - Feature importance: {feature_importance}')

    feature_importance = clf_bagging.compute_feature_importance()
    plot_feature_importance(
        feature_importance=feature_importance,
        feature_names=X_train_enc.columns,
        title="Bagging Feature Importance",
        save_path="./bagging_importance.png",
    )

    # Decision Tree
    clf_tree = DecisionTree(max_depth=7)
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = (y_pred_classes == y_test).mean()
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')

    arr = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1])
    gini_value = gini(arr)
    entropy_value = entropy(arr)

    logger.info(f"Gini index of arr = {gini_value:.6f}")
    logger.info(f"Entropy of arr = {entropy_value:.6f}")

    plot_dt_feature_importance(
        tree=clf_tree.tree,
        feature_names=X_train_enc.columns,
        save_path="./decision_tree_importance.png"
    )

    # logger.info(f"Decision Tree Feature importance: {importance}")


if __name__ == '__main__':
    main()
