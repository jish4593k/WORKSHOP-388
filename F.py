import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from tkinter import Tk, Label, Button

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def train_neural_network(X_train, y_train, input_size, hidden_size, output_size, epochs=10):
    model = NeuralNetwork(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in tqdm(range(epochs), desc="Training"):
        inputs = torch.Tensor(X_train).float()
        labels = torch.LongTensor(y_train)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return model

def evaluate_model(model, X_test, y_test):
    inputs = torch.Tensor(X_test).float()
    outputs = model(inputs)
    _, predictions = torch.max(outputs, 1)
    y_true = torch.LongTensor(y_test)
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, average='weighted')
    recall = recall_score(y_true, predictions, average='weighted')
    f1 = f1_score(y_true, predictions, average='weighted')
    return accuracy, precision, recall, f1

def main(repeats=30, seed0=5, outf="linkvsrelu.json"):
    repeats = int(repeats)
    seed0 = int(seed0)

    act_types = ["linkact", "relu", "rand", "tanh"]
    d_counts = [32, 64]

    out = {"accuracy": {}, "precision": {}, "recall": {}, "f1": {}}

    for i in range(repeats):
        seed = seed0 + i
        for d in d_counts:
            for act_type in act_types:
                uid = f"{act_type}-{d}"

                # Assuming you have data X and labels y
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

                # Train and evaluate the neural network
                model = train_neural_network(X_train, y_train, input_size=d, hidden_size=128, output_size=len(np.unique(y)))
                acc, prec, rec, f1 = evaluate_model(model, X_test, y_test)

                out["accuracy"].setdefault(uid, []).append(acc)
                out["precision"].setdefault(uid, []).append(prec)
                out["recall"].setdefault(uid, []).append(rec)
                out["f1"].setdefault(uid, []).append(f1)

    with open(outf, "w") as output_file:
        json.dump(out, output_file)

    print(f"Saved results to {outf}.")

if _name_== "__main__":
    main()

