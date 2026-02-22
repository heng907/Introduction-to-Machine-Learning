import typing as t
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layer model.
    No non-linear activation in the `intermediate layers` allowed.
    """
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        # 1 layer
        self.linear = nn.Linear(input_dim, 1) # W^T x + b


    def forward(self, x):
        """
        x: (batch_size, input_dim)
        h(x)=w^Tx+b
        """
        output = self.linear(x)     # (batch_size, 1)
        return output.squeeze(-1)    # (batch_size, )


def entropy_loss(outputs, targets):
    eps = 1e-10
    prob = torch.sigmoid(outputs) # output -> probability
    # 1: -log(p), 0: -log(1-p)
    loss = -(targets * torch.log(prob + eps) + (1 - targets) * torch.log(1 - prob + eps))
    return loss

def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    y_trues = np.asarray(y_trues)
    y_preds = [np.asarray(p) for p in y_preds]

    plt.figure(figsize=(6, 6))
    for i, p in enumerate(y_preds):
        fpr, tpr, _ = roc_curve(y_trues, p)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Learner {i}, AUC={roc_auc:.4f}')

    plt.plot([0, 1], [0, 1], 'k--')  # baseline
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC of Weak Learners')
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()

def plot_feature_importance(
    feature_importance,
    feature_names, 
    title="Feature Importance",
    save_path="./feature_importance.png"
):
    importance = np.asarray(feature_importance, dtype=float)

    original_features = [
        "person_age", "person_gender", "person_education", "person_income",
        "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
        "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
        "credit_score", "previous_loan_defaults_on_file"
    ]

    mapping = {feat: [] for feat in original_features}

    for idx, name in enumerate(feature_names):
        for orig in original_features:
            if name.startswith(orig):
                mapping[orig].append(idx)
                break

    agg_importance = []
    for orig in original_features:
        idxs = mapping[orig]
        if len(idxs) == 0:
            agg_importance.append(0.0)
        else:
            agg_importance.append(float(np.sum(np.abs(importance[idxs]))))

    # keep original order (no sorting)
    agg_importance = np.array(agg_importance)
    orig_feats = np.array(original_features)

    plt.figure(figsize=(16, 10))
    plt.barh(orig_feats, agg_importance, color="skyblue")
    plt.xlabel("Feature Importance")
    plt.ylabel("Features")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()