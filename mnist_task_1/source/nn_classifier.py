import torch
import torch.nn as nn

from .interface import MnistClassifierInterface

class FeedForwardNN(MnistClassifierInterface):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
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
        
        
    def train(self, X_train, y_train, num_epochs=5, batch_size=64, learning_rate=0.001):
        X_train = self.flatten(X_train).float()
        y_train = y_train.long()  # Ensure labels are of type long for CrossEntropyLoss
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            for images, labels in dataloader:
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
    def predict(self, X_test):
        X_test = self.flatten(X_test).float()
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs.data, 1)
        return predicted
 