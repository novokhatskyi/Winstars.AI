from .rf_classifier import RandomForestMnistClassifier
from .nn_classifier import FeedForwardNN
from .cnn_classifier import CNNClassifier
'''
algorithm: 
"rf" - Random Forest Classifier
"nn" - Neural Network Classifier
"cnn" - Convolutional Neural Network Classifier
'''

class MnistClassifier:
    def __init__(self, algorithm="rf"):
        if algorithm == "rf":
            self.model = RandomForestMnistClassifier()
        elif algorithm == "nn":
            self.model = FeedForwardNN()
        elif algorithm == "cnn":
            self.model = CNNClassifier()
        else:
            raise ValueError(f"Invalid algorithm choice. Please choose 'rf', 'cnn', or 'nn' {algorithm}")

    def train(self, X_train, y_train, **kwargs):
        print(f"Training {self.model.__class__.__name__}...")
        self.model.train(X_train, y_train, **kwargs)

    def predict(self, X_test):
        print(f"Making predictions with {self.model.__class__.__name__}...")
        return self.model.predict(X_test)