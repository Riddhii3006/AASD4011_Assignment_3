import torch
from torch import nn


def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    """
    model = nn.Linear(input_size, output_size)
    return model


def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def fit_regression_model(X, y):
    """
    Train the model for the given number of epochs.
    """
    learning_rate = 0.001  # Lowered learning rate
    num_epochs = 5000  # Number of epochs, adjust based on convergence
    input_features = X.shape[1]  # Extract the number of features from X
    output_features = y.shape[1]  # Extract the number of features from y
    model = create_linear_regression_model(input_features, output_features)

    loss_fn = nn.MSELoss()  # Using mean squared error loss

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    prev_loss = float("inf")

    for epoch in range(num_epochs):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss {loss.item()}")
        if abs(prev_loss - loss.item()) < 1e-6:  # Stopping condition
            break
        prev_loss = loss.item()

    return model, loss