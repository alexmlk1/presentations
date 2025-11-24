import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim 
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Initialize global variables
np.random.seed(42)  # reproducibility
n = 20
d_in = 2
d_out = 1
LOO_L = []  #losses on the two models with LOO
IF_L = []   #losses on the two models with IFs
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Generate sample data (20 samples, 2 features)
X = np.random.randn(n, d_in).astype(np.float32)   # feature matrix (20 × 2)
e = np.random.normal(0, 0.1, n).astype(np.float32)  # data error
y = (X[:, 0] + 0.3*X[:, 1] -0.5 + e[:] > 0).astype(np.float32)  # binary labels based on a simple rule


# Define MLP
class SingleHiddenMLP(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, 10, bias=False)   # Hidden layer (2 → 10)
        self.relu = nn.Tanh()
        self.fc2 = nn.Linear(10, d_out, bias=False)   # Output layer (10 → 1)
        self.sigmoid = nn.Sigmoid()   #sigmoid cud we need a probability, otherwise linear

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def train_MLP(X,y,MAX_IT=500):
    global device

    model = SingleHiddenMLP(d_in,d_out)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(MAX_IT):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    #print(f"Epoch {epoch+1}, Loss = {loss.item():.5f}")

    return model


def train_LR(X,y):
    """Train Logistic Regression model"""
    model = LogisticRegression()
    model.fit(X, y)
    return model


def train_lR(X,y):
    """Train Linear Regression model"""
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_models(X,y):
    # Convert data from np/sk to torch
    X_t = torch.tensor(X)
    y_t = torch.tensor(y).view(-1, 1)  #-1: crea con le dimensioni giuste

    return train_LR(X,y), train_MLP(X_t,y_t)


def criterion(z): return sum([(a-b)**2/2 for (a,b) in z])/len(z)


def eval_models(X,y,L):
    # Train-test split
    #X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=0.3, random_state=42
    #)
    lR,M = train_models(X,y)

    # Predictions
    preds_lR = lR.predict(X)
    #print("\nPredictions:", preds_lR)
    #print("True labels:", y)

    with torch.no_grad():
        preds_M = (M(torch.tensor(X)) > 0.5).float().squeeze()
    #print("\nPredictions:", preds_M.numpy())
    #print("True labels:", y_test)

    L.append([criterion(list(zip(preds_lR, y))).item(), criterion(list(zip(preds_M, y))).item()])
    return L


def eval_LOO(X,y):
    global LOO_L
    eval_models(X,y,LOO_L)
    for index in range(len(X)):
        X_new=np.delete(X,index,0)
        y_new=np.delete(y,index,0)
        eval_models(X_new,y_new,LOO_L)
    return LOO_L


def L_IF_lR(i, X, y):
    H = np.matrix(X.transpose() @ X)
    g = X[i]*(y[i]-(X[i].transpose())
    @((np.linalg.inv(H))@ X.transpose())@y.transpose()).item()
    return g.transpose() @ np.linalg.solve(H,g)


def L_IF_MLP(i, X, y):
    X = torch.tensor(X)
    y = torch.tensor(y).view(-1, 1)
    M = train_MLP(X,y)
    # Extract shapes and sizes of parameters
    params = [p for p in M.parameters()]
    shapes = [p.shape for p in params]
    sizes  = [p.numel() for p in params]

    # Create flat parameter vector (this will be optimized/differentiated)
    theta0 = torch.cat([p.detach().flatten() for p in params]).clone().requires_grad_(True)

    # Function to turn theta into a list of tensors with correct shapes
    def vector_to_params(theta):
        out = []
        idx = 0
        for shape, size in zip(shapes, sizes):
            out.append(theta[idx:idx+size].view(shape))
            idx += size
        return out

    # ---- functional forward pass using your module architecture ----
    def functional_forward(theta):
        W1, W2 = vector_to_params(theta)  # fc1.weight, fc2.weight
        h = torch.tanh(X @ W1.t())
        out = torch.sigmoid(h @ W2.t())
        return out
    
    external_grad = torch.tensor([1.]*n)
    external_grad = external_grad.view(-1, 1)
    evaluation = functional_forward(theta0)
    evaluation.backward(gradient=external_grad)
    g = np.sqrt((sum([(a-b)**2/2 for (a,b) in list(zip(M(X),y))])).detach().numpy()).item()*theta0.grad

    # ---- loss as a function of theta ----
    def loss_fn(theta):
        preds = functional_forward(theta)
        return criterion([a,b for (preds, y)])

    # Compute Hessian
    H = torch.autograd.functional.hessian(loss_fn, theta0).detach().numpy()
    g = g.detach().numpy()

    #print("Is Hessian zero?", torch.all(H == 0).item())

    return g.transpose() @ np.linalg.solve(H,g)


def visualize_result(title, IFL, LOOL):
    plt.close("all")
    max_abs = np.max([np.abs(IFL), np.abs(LOOL)])
    min_, max_ = -max_abs * 1.1, max_abs * 1.1
    plt.rcParams['figure.figsize'] = 6, 5
    plt.scatter(IFL.pop(0), LOOL.pop(0), zorder=2, s=10, c='red') #error without taking out any point
    plt.scatter(IFL, LOOL, zorder=2, s=10)
    plt.title(title)
    plt.xlabel('IF loss diff')
    plt.ylabel('LOO loss diff')
    range_ = [min_, max_]
    plt.plot(range_, range_, 'k-', alpha=0.2, zorder=1)  #retta (zorder=1 vuol dire che viene stampata per prima)
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.savefig("high_res_figure.png", dpi=600)
    plt.show()

if __name__ == '__main__':
    Z = eval_LOO(X,y)

    #visualize_result('IF vs LOO (Linear Regression)', [Z[i][0] for i in range(len(X))], [L_IF_lR(i,X,y) for i in range(len(X))])
    visualize_result('IF vs LOO (MLP)', [Z[i][1] for i in range(len(X))], [L_IF_MLP(i,X,y) for i in range(len(X))])