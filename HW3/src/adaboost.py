import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        """Free to add args as you need, like batch-size, learning rate, etc."""

        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []
        self.input_dim = input_dim

    def fit(self, X_train, y_train, num_epochs: int = 700, learning_rate: float = 0.002):
        """
        TODO: Implement the training part
        """

        """
        X_train: (N, D) numpy
        y_train: (N,) numpy, values in {0,1}
        """

        # convert to tensor
        X_train = torch.tensor(X_train, dtype=torch.float32) # features
        y_train = torch.tensor(y_train, dtype=torch.float32) # labels
        
        n_samples = X_train.shape[0]
        # transition {0, 1} -> {-1, +1}
        trans_y_train = y_train.clone()
        trans_y_train[trans_y_train == 0] = -1.0

        # initialize weak learners
        w = torch.ones(n_samples) / n_samples
        self.sample_weights = w.clone().numpy()

        for learner in self.learners:
            optimizer = optim.SGD(learner.parameters(), lr=learning_rate)
            # ======== train ========
            for _ in range(num_epochs):
                optimizer.zero_grad()
                logits = learner(X_train) # (N,) row output
                per_sample_loss = entropy_loss(logits, y_train)
                # weighted loss：sum_i w_i * loss_i / sum_i w_i
                loss = torch.sum(w*per_sample_loss)/torch.sum(w)
                loss.backward()
                optimizer.step()

            # === calculate weighted error ===
            with torch.no_grad():
                logits = learner(X_train) # raw output
                probs = torch.sigmoid(logits) # raw output -> prob
                y_pred = (probs >= 0.5).float()

                # transition {0, 1} -> {-1, 1}
                trans_y_pred = y_pred.clone()
                trans_y_pred[trans_y_pred == 0] = -1.0

                misclassified = (trans_y_pred != trans_y_train)
                # et​=sigma_i (w_i​⋅1/[ht​(x_i​)!=y_i​])
                errors = torch.sum(w * misclassified.float()) / torch.sum(w)
                errors = torch.clamp(errors, 1e-10, 1 - 1e-10) # avoid from error = 0 or error = 1

                # calculate alpha
                alpha = 0.5 * torch.log((1 - errors)/errors)
                alpha_val = alpha.item()
                self.alphas.append(alpha_val)

                # update sample weights
                w = w * torch.exp(-alpha * trans_y_train * trans_y_pred)
                # normalize weight
                w = w / torch.sum(w)

                self.sample_weights = w.clone().numpy()

        return self

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """
        TODO: Implement the prediction
        """
        X = torch.tensor(X, dtype=torch.float32)
        n_samples = X.shape[0]
        # to save every learners predict probs
        learners_probs = []
        # F(x_i​)=sigma_t (​alpha_t * ​h_t​(x_i​))
        final_score = np.zeros(n_samples, dtype=np.float64)

        with torch.no_grad():
            for alpha, learner in zip(self.alphas, self.learners):
                logits = learner(X)
                probs = torch.sigmoid(logits).numpy()
                learners_probs.append(probs)

                # trans {0, 1} to {-1, 1}
                trans_probs = np.where(probs >= 0.5, 1.0, -1.0)
                final_score += alpha * trans_probs

        # ==== decide class ====
        trans_final = np.sign(final_score)
        """
        final_score[i]>0: trans_final[i] = +1
        final_score[i]=0: trans_final[i] = 0
        final_score[i]<1: trans_final[i] = -1
        """
        trans_final[trans_final == 0] = 1
        # converse {-1, 1} -> {0,1}
        y_pred_class = np.where(trans_final > 0, 1, 0)
        return y_pred_class, learners_probs

    def compute_feature_importance(self) -> t.Sequence[float]:
        """
        TODO: Implement the feature importance calculation
        """

        # feature_importance_j = sum_t alpha_t * |w_j^(t)|
        if not self.alphas:
            return []
        n_features = self.input_dim
        importance = np.zeros(n_features, dtype=np.float64)

        for alpha, learner in zip(self.alphas, self.learners):
            # 1-layer linear
            w = learner.linear.weight.detach().numpy().reshape(-1)  # (D,)
            importance += alpha * np.abs(w)
        
        return importance.tolist()