# -*- coding: utf-8 -*-

"""
Created on 30. 05. 2025 at 15:08:33

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Correlations Analysis
"""
import torch
from torch import nn
from torch import optim
# from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from analyze import VariableCorrelationAnalyzer
from utils.config_singleton import ConfigSingleton
# from data_loader import TabularDataset
from splitter import TrainTestSplitter
from mlp import MLP

ConfigSingleton.set()
VariableCorrelationAnalyzer().pipeline()
train, test = TrainTestSplitter().get()

# dataset_train = TabularDataset(train)
# dataset_test = TabularDataset(test)

model = MLP()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.BCELoss()  # Binary classification loss
optimizer = optim.Adam(
    model.parameters(),
    lr = ConfigSingleton.get().model.training.lr,
    )
# Training loop
n_epochs = ConfigSingleton.get().model.training.epochs

for epoch in range(n_epochs):
    model.train()
    permutation = torch.randperm(train.shape[0])  # zamíchání indexů

    EPOCH_LOSS = 0.0
    for i in range(0, train.shape[0], ConfigSingleton.get().model.training.batch_size):
        indices = permutation[i:i + ConfigSingleton.get().model.training.batch_size]
        batch_X = torch.Tensor(train[indices,:-1])
        batch_y = torch.Tensor(train[indices, -1:])

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        EPOCH_LOSS += loss.item()
        EPOCH_LOSS /= (train.shape[0] // ConfigSingleton.get().model.training.batch_size)


    # Eval po každé epoše
    model.eval()
    with torch.no_grad():
        y_test_pred = model(torch.Tensor(test[:,:-1]))
        test_loss = criterion(y_test_pred, torch.Tensor(test[:,-1:]))
    if epoch % 20 == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch+1}/{n_epochs},"\
            f"Train Loss: {EPOCH_LOSS:.4f}, Test Loss: {test_loss.item():.4f}")


y_test_pred = (y_test_pred > 0.5).float()  # Thresholding for binary classification
f1 = f1_score(test[:,-1], y_test_pred.numpy(), average='binary')
print(f"F1 Score: {f1:.4f}")
