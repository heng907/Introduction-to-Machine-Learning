"""
1. Complete the implementation for the `...` part
2. Feel free to take strategies to make faster convergence
3. You can add additional params to the Class/Function as you need. But the key print out should be kept.
4. Traps in the code. Fix common semantic/stylistic problems to pass the linting
"""

from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        """Question1
        Complete this function
        """
        # beta hat = (X^T * X)^(-1) * X^T * y
        # beta_0 = intercept, beta_1 = weights(slope)
        X = np.c_[np.ones((X.shape[0], 1)), X]
        BetaHat = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.intercept = BetaHat[0]
        self.weights = BetaHat[1:]

    def predict(self, X):
        """Question4
        Complete this function
        """
        pred = X @ self.weights + self.intercept
        return pred


class LinearRegressionGradientdescent:
    def fit(
        self,
        X,
        y,
        learning_rate: float,
        epochs: float
    ):
        """Question2
        Complete this function
        """
        # initialize
        lr = learning_rate  # learning rate
        N, M = X.shape
        self.weights = np.zeros(M)
        self.intercept = 0.0

        # normalized
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - mean) / std

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        losses, lr_history = [], []
        epochs = int(epochs)  # range() must be int
        for epoch in range(epochs):
            # y_pred: the prediction of y = beta_0 + beta_1 * x; y: is the ground truth
            y_pred = X @ self.weights + self.intercept
            loss = compute_mse(y_pred, y)

            losses.append(loss)
            lr_history.append(lr)

            # calculate gradient
            r = y_pred - y  # r: residual
            grad_w = (2.0 / N) * (X.T @ r)  # grab_w: gradient of weight
            grad_b = (2.0 / N) * np.sum(r)  # grab_b: gradient of bias

            # update
            self.weights -= lr * grad_w
            self.intercept -= lr * grad_b

            if epoch % 1000 == 0:
                logger.info(f'EPOCH {epoch}, {loss=:.4f}, {lr=:.4f}')
        w_temp = self.weights.copy()
        self.weights = w_temp / std
        self.intercept = self.intercept - np.sum(w_temp * (mean / std))

        return losses, lr_history

    def predict(self, X):
        """Question4
        Complete this
        """
        pred = X @ self.weights + self.intercept
        return pred


def compute_mse(prediction, ground_truth):
    mse = np.mean((prediction - ground_truth) ** 2)
    return mse


def main():
    train_df = pd.read_csv('./train.csv')  # Load training data
    test_df = pd.read_csv('./test.csv')  # Load test data
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)

    """This is the print out of question1"""
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    # here
    losses, lr_history = LR_GD.fit(train_x, train_y, learning_rate=0.015, epochs=300)

    """
    This is the print out of question2
    Note: You need to screenshot your hyper-parameters as well.
    """
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')
    """
    Question3: Plot the learning curve.
    Implement here
    """
    loss = plt.plot(losses)
    plt.title("Training loss")
    plt.legend(loss, ["Train MSE loss"])
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.show()

    """Question4"""
    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
