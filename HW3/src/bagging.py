import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, entropy_loss


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        """Free to add args as you need, like batch-size, learning rate, etc."""

        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]
        self.input_dim = input_dim

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """
        TODO: Implement the training part
        """
        # convert to tensor
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        
        n_samples = X_train.shape[0]
        for learner in self.learners:
            optimizer = optim.SGD(learner.parameters(), lr=learning_rate)

            # === bootstrap sampling ===
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X_train[indices]
            y_bootstrap = y_train[indices]

            # === train ===
            for _ in range(num_epochs):
                optimizer.zero_grad()

                logits = learner(X_bootstrap) # row output
                # === calculate loss ===
                per_sample_loss = entropy_loss(logits, y_bootstrap)
                loss = per_sample_loss.mean() # bagging no weight

                loss.backward()
                optimizer.step()

        return self

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """
        TODO: Implement the training part
        """
        # convert to tensor
        X = torch.tensor(X, dtype=torch.float32)
        n_samples = X.shape[0]

        learners_probs = []

        with torch.no_grad():
            for learner in self.learners:
                logits = learner(X)
                probs = torch.sigmoid(logits).numpy()
                learners_probs.append(probs)

        # put probs together -> (num_learners, N)
        probs_stack = np.stack(learners_probs, axis=0)
        # mean every samples -> (N, )
        avg_probs = probs_stack.mean(axis=0)

        """
        avg >= 0.5 ,then class 1
        avg < 0.5  ,then class 0
        """
        y_pred_class = (avg_probs >= 0.5).astype(int)
        
        return y_pred_class, learners_probs

    def compute_feature_importance(self) -> t.Sequence[float]:
        """
        TODO: Implement the feature importance calculation
        """
        # feature_importance_j = sum_t |w_j^(t)|
        n_features = self.input_dim
        importance = np.zeros(n_features, dtype=np.float64)

        for learner in self.learners:
            # 1-layer linear
            w = learner.linear.weight.detach().cpu().numpy().reshape(-1)  # (D,)
            importance += np.abs(w)
            
        return importance.tolist()


