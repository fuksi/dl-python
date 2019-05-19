import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

device = torch.device("cpu")
np.random.seed(123)
torch.manual_seed(123)

data = np.genfromtxt('dataset.csv', delimiter=',', skip_header=1)
x = np.asarray([datapoint[1:12] for datapoint in data])
y = np.asarray([datapoint[-1] for datapoint in data]) # last column is the target

# Let us split the data into training, validation and test sets and plot the training and validation sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, shuffle=True)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1, shuffle=True)

print(x_train.shape)
print(y_train.shape)

"""
Single layer perceptron
"""
class SLP(nn.Module):
    def __init__(self):
        super(SLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(11, 3))
            
    def forward(self, x):
        return self.net(x)

"""
Multilayer perception
In the code below, we define a neural network architecture with:
* input dimension 12
* one hidden layer with 100 units with tanh nonlinearity
* tanh nonlinearity
* one hidden layer with 100 units with tanh nonlinearity
* linear output layer with output dimension 3
"""

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 3)
        )
            
    def forward(self, x):
        return self.net(x)

"""
    Utility function
"""
def compute_loss(mlp, x, y):
    mlp.eval()
    with torch.no_grad():
        x = torch.tensor(x, device=device, dtype=torch.float)
        y = torch.tensor(y, device=device, dtype=torch.int64)
        outputs = mlp.forward(x)
        loss = nn.CrossEntropyLoss()(outputs, y)
        return np.asscalar(loss.cpu().data.numpy())

def print_progress(epoch, train_error, val_error):
    print('Train Epoch {}: Train error {:.2f} Validation error {:.2f}'.format(
        epoch, train_error, val_error))


def train(network, network_name):
    mlp = MLP()
    mlp.to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
    n_epochs = 10000
    train_errors = []  # Keep track of the training data
    val_errors = []  # Keep track of the validation data
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        x = torch.tensor(x_train, device=device, dtype=torch.float)
        y = torch.tensor(y_train, device=device, dtype=torch.int64)

        optimizer.zero_grad()
        outputs = mlp.forward(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch % 500) == 0:
            train_errors.append(compute_loss(mlp, x_train, y_train))
            val_errors.append(compute_loss(mlp, x_val, y_val))
            print_progress(epoch, train_errors[-1], val_errors[-1])

    test_loss_no_regularization = compute_loss(mlp, x_test, y_test)
    print("Test loss without regularization: %.5f" % test_loss_no_regularization)

    fig, ax = plt.subplots(1)
    ax.loglog(train_errors)
    ax.loglog(val_errors)
    plt.savefig(network_name)


# mlp = MLP()
# mlp.to(device)
# train(mlp, 'mlp')


