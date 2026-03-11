from sklearn.ensemble import RandomForestClassifier
from .interface import MnistClassifierInterface

# Class that implements the Random Forest classifier for MNIST dataset
class RandomForestMnistClassifier(MnistClassifierInterface):
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

    # Helper method to flatten the input data
    def flatten(self, X):
        if X.ndim == 2: #[N, 784]
            return X
        elif X.ndim == 3: #[N, 28, 28]
            return X.reshape((X.shape[0], -1))
        elif X.ndim == 4: #[N, 1, 28, 28]
            return X.reshape((X.shape[0], -1))
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")

    def train(self, X_train, y_train, **kwargs):
        X_train = self.flatten(X_train)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        X_test = self.flatten(X_test)
        return self.model.predict(X_test)