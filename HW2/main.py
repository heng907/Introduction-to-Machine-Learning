import typing as t
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        # use Gradient Descent + Cross-Entropy Loss to find weights(w) and intercept(b).
        # initial, set N rows and M cols
        N, M = inputs.shape
        self.weights = np.zeros(M)
        self.intercept = 0.0

        # calculate gradient
        for i in range(self.num_iterations):
            y_pred = self.sigmoid(inputs @ self.weights + self.intercept)
            # dw = 1/N * X^T.(p-y), db = 1/N * sig(p-y)
            grad_w = (inputs.T @ (y_pred - targets)) / N    # grad_w: gradient of weight
            grad_b = (np.sum(y_pred - targets)) / N         # grad_b: gradient of bias
            # update
            self.weights -= self.learning_rate * grad_w
            self.intercept -= self.learning_rate * grad_b
            # loss function(CE): L = (-1/N)*sig[y_i*log(p_i) + (1 - y_i)log(1 - p_i)]
            if (i % 200 == 0):
                eps = 1e-15
                loss = -np.mean(targets * np.log(y_pred) + (1 - y_pred) * np.log(1 - y_pred + eps))
                print(f'Iteration {i}: Loss= {float(loss):.4f}')

        # raise NotImplementedError

    def predict(self, inputs: npt.NDArray[float],) -> t.Tuple[t.Sequence[float], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        # use weights(w) and intercept(b) to predict prob. of class_1(pred_prob) and predicted class(pred_class)
        pred_prob = self.sigmoid(inputs @ self.weights + self.intercept)
        pred_class = np.where(pred_prob >= 0.5, 1, 0)
        return pred_prob, pred_class
        # raise NotImplementedError

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        # def of sigmoid function: sig(x) = 1/1+exp(-x)
        return 1 / (1 + np.exp(-x))
        # raise NotImplementedError


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        # J(W) = (m_2 - m_1)^2 / (s_1)^2 + (s_2)^2

        # mean vector m_0 = 1/N_0 * sig(x_i), m_1 = 1/N_1 * sig(x_i)
        x_0 = inputs[targets == 0]
        x_1 = inputs[targets == 1]
        self.m0 = np.mean(x_0, axis=0)
        self.m1 = np.mean(x_1, axis=0)

        # within-class matrix: S_W = sig(x_1 - m1)(x_1 - m1)^T + sig(x_2 - m2)(x_2 - m2)^T
        self.sw = (x_0 - self.m0).T @ (x_0 - self.m0) + (x_1 - self.m1).T @ (x_1 - self.m1)
        # between-class matrix: S_B = (m_2 - m_1)(m_2 - m_1)^T
        diff = (self.m1 - self.m0)
        self.sb = np.outer(diff, diff)
        # weight: w = S_w^-1 * (m_1 - m_0)
        self.w = np.linalg.inv(self.sw) @ (self.m1 - self.m0)
        # slope, for sketch
        self.slope = self.w[1] / self.w[0]

        proj_m0 = self.m0 @ self.w
        proj_m1 = self.m1 @ self.w
        self.threshold = (proj_m0 + proj_m1) / 2
        self.class1_is_larger = (proj_m1 > proj_m0)
        # raise NotImplementedError

    def predict(self, inputs: npt.NDArray[float],) -> t.Sequence[t.Union[int, bool]]:
        # predict test cases close to m0(pred_0) or m1(pred_1)
        # t = x^T * w

        proj_inputs = inputs @ self.w

        if self.class1_is_larger:
            return np.where(proj_inputs >= self.threshold, 1, 0)
        else:
            return np.where(proj_inputs < self.threshold, 1, 0)

        # raise NotImplementedError

    def plot_projection(self, inputs: npt.NDArray[float], targets: t.Sequence[int],):
        y_true = np.array(targets)
        y_pred = self.predict(inputs)

        plt.figure(figsize=(10, 10))
        correct_mask = (y_pred == y_true)

        legend_labels = set()

        # Plot correct predictions (green)
        for i in [0, 1]:  # For each class
            marker = 'o' if i == 0 else '^'
            mask_correct = (y_true == i) & correct_mask
            if np.any(mask_correct):
                label = f'Correct Class {i}'
                plt.scatter(inputs[mask_correct, 0], inputs[mask_correct, 1], color='green', marker=marker,
                            label=label if label not in legend_labels else "", zorder=3)
                legend_labels.add(label)
            # Plot incorrect predictions (red)
            mask_incorrect = (y_true == i) & ~correct_mask
            if np.any(mask_incorrect):
                label = f'Incorrect Class {i}'
                plt.scatter(inputs[mask_incorrect, 0],
                            inputs[mask_incorrect, 1],
                            color='red',
                            marker=marker,
                            label=label if label not in legend_labels else "",
                            zorder=3)
                legend_labels.add(label)
        # plot range
        x_min, x_max = inputs[:, 0].min() - 1, inputs[:, 0].max() + 1
        y_min, y_max = inputs[:, 1].min() - 1, inputs[:, 1].max() + 1

        # Plot projection line (gray)
        mid_point_2d = (self.m0 + self.m1) / 2

        y_proj_min = self.slope * (x_min - mid_point_2d[0]) + mid_point_2d[1]
        y_proj_max = self.slope * (x_max - mid_point_2d[0]) + mid_point_2d[1]
        plt.plot([x_min, x_max],
                 [y_proj_min, y_proj_max],
                 'gray',
                 label='Projection Line',
                 alpha=0.7,
                 zorder=1)

        # Plot decision boundary (blue)
        x_vals_decision = np.array([x_min, x_max])
        if np.abs(self.w[1]) > 1e-8:
            y_vals_decision = (self.threshold - self.w[0] * x_vals_decision) / self.w[1]
            plt.plot(x_vals_decision,
                     y_vals_decision,
                     'b-',
                     label='Decision Boundary',
                     zorder=2)
        else:
            x_val_vertical = self.threshold / self.w[0]
            plt.vlines(x_val_vertical,
                       ymin=y_min,
                       ymax=y_max,
                       colors='blue',
                       linestyles='solid',
                       label='Decision Boundary',
                       zorder=2)
        # Project points onto 1D space
        norm_w_sq = np.linalg.norm(self.w)**2
        if norm_w_sq < 1e-8:
            norm_w_sq = 1e-8

        proj_points_2d = np.empty_like(inputs)
        for i in range(len(inputs)):
            p = inputs[i]
            p_proj = mid_point_2d + (np.dot(p - mid_point_2d, self.w) / norm_w_sq) * self.w
            proj_points_2d[i] = p_proj
            plt.plot([p[0], p_proj[0]],
                     [p[1], p_proj[1]],
                     color='gray',
                     linestyle=':',
                     alpha=0.5,
                     zorder=0)

        plt.scatter(proj_points_2d[:, 0], proj_points_2d[:, 1], c='black', marker='.', s=15,
                    label='Projected Points', zorder=4)
        plt.title(f'Projection onto FLD axis (w=[{self.w[0]:.5f}, {self.w[1]:.5f}])')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.show()

        # raise NotImplementedError


def compute_auc(y_trues, y_preds):
    return roc_auc_score(y_trues, y_preds)
    # raise NotImplementedError


def accuracy_score(y_trues, y_preds):
    correct_pred = np.sum(y_trues == y_preds)
    total_pred = len(y_trues)
    return correct_pred / total_pred
    # raise NotImplementedError


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=0.05,  # You can modify the parameters as you want
        num_iterations=1500,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    """
    (TODO): Implement your code to
    1) Fit the FLD model
    2) Make prediction
    3) Compute the evaluation metrics

    Please also take care of the variables you used.
    """
    FLD_.fit(x_train, y_train)
    y_pred = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """

    FLD_.plot_projection(x_test, y_test)


if __name__ == "__main__":
    main()
