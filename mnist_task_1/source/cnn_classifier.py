from .interface import MnistClassifierInterface
import torch
import torch.nn as nn


class CNNClassifier(MnistClassifierInterface):
    def __init__(self, num_classes=10):
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
            nn.Linear(64 * 10 * 10, num_classes)
        )
    
    def reshape_input(self, X):
        if X.ndim == 2: #[N, 784]
            return X.view(-1, 1, 28, 28)
        elif X.ndim == 3: #[N, 28, 28]
            return X.view(-1, 1, 28, 28)
        elif X.ndim == 4: #[N, 1, 28, 28]
            return X.view(-1, 1, 28, 28)
        else:
            raise ValueError(f"Unexpected input shape: {X.shape}")
        
    def train(self, X_train, y_train, num_epochs=5, batch_size=64, learning_rate=0.001):
        X_train = self.reshape_input(X_train).float()
        y_train = y_train.long() 

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
        X_test = self.reshape_input(X_test).float()

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test)
            _, predicted = torch.max(outputs, 1)
        return predicted