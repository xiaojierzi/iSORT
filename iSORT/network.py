import scanpy as sc
import numpy as np
import random
import torch
import warnings
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNet(nn.Module):
    """
    Function: A neural network class implementing a multi-layer perceptron with dropout.

    This class creates a neural network with a specified number of hidden layers, each 
    followed by a ReLU activation and dropout. The final layer is a linear layer without 
    activation.

    Parameters:
    - input_features (int): The number of input features.
    - hidden_layers (list of int): List containing the sizes of each hidden layer.
    - output_features (int): The number of output features.

    """

    def __init__(self, input_features, hidden_layers, output_features, use_dropout=True):
        super(NeuralNet, self).__init__()
        self.use_dropout = use_dropout
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_features, hidden_layers[0]))

        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))

        self.layers.append(nn.Linear(hidden_layers[-1], output_features))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = nn.ReLU()(self.layers[i](x))
            if self.use_dropout:
                x = self.dropout(x)
        x = self.layers[-1](x) 
        return x



def initialize_model(input_features, hidden_layers, output_features, use_dropout=True):
    """
    Function: Initializes and returns a neural network model, loss criterion, and optimizer.

    Parameters:
    - input_features (int): The number of input features for the network.
    - hidden_layers (list of int): Sizes of hidden layers.
    - output_features (int): The number of output features for the network.

    Outputs:
    - model (NeuralNet): Initialized neural network model.
    - criterion (nn.MSELoss): Mean squared error loss function.
    - optimizer (optim.Adam): Adam optimizer for the model.

    """

    model = NeuralNet(input_features, hidden_layers, output_features,use_dropout).to('cuda')
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    return model, criterion, optimizer


def prepare_data(source_mapped, spatial, weights=None, if_weighted=True):
    """
    Function: Prepares data loaders for training and validation.

    The function splits the data into training and validation sets, wraps them in 
    TensorDataset, and returns DataLoaders for each set.

    Parameters:
    - source_mapped (np.array): Source data array.
    - spatial (np.array): Target spatial data array.
    - weights (np.array): Array of weights for each data point.

    Outputs:
    - train_loader (DataLoader): DataLoader for training data.
    - val_loader (DataLoader): DataLoader for validation data.

    """
    if if_weighted:
        X = torch.tensor(source_mapped, dtype=torch.float32).to('cuda')
        Y = torch.tensor(spatial, dtype=torch.float32).to('cuda')
        W = torch.tensor(weights, dtype=torch.float32).to('cuda')
        dataset = TensorDataset(X, Y, W)
    else:
        X = torch.tensor(source_mapped, dtype=torch.float32).to('cuda')
        Y = torch.tensor(spatial, dtype=torch.float32).to('cuda')
        dataset = TensorDataset(X, Y)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    return train_loader, val_loader


def seed_everything(seed=1234):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def predict_and_plot(model, test_data, adata, cluster_name):
    """
    Function: Predicts using the model and plots the results in an embedding space.

    The function performs predictions on the test data, stores the results in the
    Anndata object, and plots the embeddings using Scanpy.

    Parameters:
    - model (NeuralNet): Trained neural network model.
    - test_data (torch.Tensor): Test data to be used for prediction.
    - adata (AnnData): Anndata object to store the prediction results.
    - cluster_name (str): Name of the variable used to color the plot points.

    Outputs:
    - predictions_array (np.array): Array of predictions from the model.

    """
    with torch.no_grad():
        test = test_data.to('cuda')
        predictions_normalized = model(test)
    predictions_normalized = predictions_normalized.to('cpu')
    predictions_array = predictions_normalized.numpy()
    
    adata.obsm['prediction_space'] = predictions_array

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.pl.embedding(adata, basis='prediction_space', color=cluster_name)
    
    return predictions_array



def compute_weighted_loss(outputs, targets, weights,criterion):
    """
    Function: Computes a weighted loss between model outputs and targets.

    Parameters:
    - outputs (torch.Tensor): Model outputs.
    - targets (torch.Tensor): Target values.
    - weights (torch.Tensor): Weights for each data point.
    - criterion (nn.MSELoss or similar): Loss function.

    Outputs:
    - Weighted mean loss.

    """

    loss = criterion(outputs, targets)
    return (loss * weights).mean()


def custom_loss(y_pred, y_true, alpha, beta_val, ns):
    """
    Function: Custom loss function based on weighted mean squared error.

    Parameters:
    - y_pred (torch.Tensor): Predicted values.
    - y_true (torch.Tensor): True values.
    - alpha (torch.Tensor): Weights tensor.
    - beta_val (float or torch.Tensor): Beta value for the loss computation.
    - ns (int or torch.Tensor): Sample size for normalization.

    Outputs:
    - Computed custom loss.

    """
   
    if isinstance(beta_val, int) or isinstance(beta_val, float):
        beta_val = torch.tensor(beta_val, dtype=torch.float32, device='cuda:0')
    if isinstance(ns, int) or isinstance(ns, float):
        ns = torch.tensor(ns, dtype=torch.float32, device='cuda:0')
    
    beta_expanded = beta_val.unsqueeze(0).unsqueeze(1)
    ns_expanded = ns.unsqueeze(0).unsqueeze(1)
    
    alpha_expanded = alpha.unsqueeze(1)
    
    loss = torch.sum(beta_expanded/ns_expanded * alpha_expanded * (y_true - y_pred) ** 2)
    return loss


def train_and_validate(model, train_loader, val_loader, optimizer, criterion, epochs=200, if_weighted=True):
    """
    Function: Trains and validates the neural network model.

    This function trains the model for a specified number of epochs and validates it 
    after each epoch, reporting the loss.

    Parameters:
    - model (NeuralNet): The neural network model to train.
    - train_loader (DataLoader): DataLoader for training data.
    - val_loader (DataLoader): DataLoader for validation data.
    - optimizer (optim.Adam or similar): Optimizer for the model.
    - criterion (loss function): Loss function to use for training.
    - epochs (int, optional): Number of training epochs.

    """

    print("Starting")
    if if_weighted:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            
            for batch_X, batch_Y, batch_weights in train_loader:
                batch_X, batch_Y, batch_weights = batch_X.to('cuda'), batch_Y.to('cuda'), batch_weights.to('cuda')
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = compute_weighted_loss(outputs, batch_Y, batch_weights, criterion)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_Y, batch_weights in val_loader:
                    batch_X, batch_Y, batch_weights = batch_X.to('cuda'), batch_Y.to('cuda'), batch_weights.to('cuda')
                    outputs = model(batch_X)
                    loss = compute_weighted_loss(outputs, batch_Y, batch_weights, criterion)
                    val_loss += loss.item()
        print('Done!')
    else:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            
            for batch_X, batch_Y in train_loader:
                batch_X, batch_Y = batch_X.to('cuda'), batch_Y.to('cuda')
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_Y in val_loader:
                    batch_X, batch_Y = batch_X.to('cuda'), batch_Y.to('cuda')
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_Y)
                    val_loss += loss.item()
        print('Done!')



def train_model(model, optimizer, EPOCHS, data_sets, labels, weights, beta_values):
    """
    Function: Trains a neural network model on multiple data sets.

    Trains the model on multiple datasets, each with its own labels, weights, and 
    beta values. Adjusts model parameters based on the custom loss function.

    Parameters:
    - model (NeuralNet): Neural network model to be trained.
    - optimizer (torch.optim): Optimizer for the neural network.
    - EPOCHS (int): Number of epochs for training.
    - data_sets (list of Anndata): List of Anndata objects containing the data.
    - labels (list of arrays): List of label arrays corresponding to each data set.
    - weights (list of arrays): List of weight arrays for each data set.
    - beta_values (list of floats): List of beta values for each data set.

    """

    for epoch in range(EPOCHS):
        total_loss = 0

        for data_set, label, weight, beta_val in zip(data_sets, labels, weights, beta_values):
            optimizer.zero_grad() 

            ns = data_set.shape[0]

            xs = torch.tensor(data_set.X, dtype=torch.float32).to(device)
            ys = torch.tensor(label, dtype=torch.float32).to(device)
            weights_tensor = torch.tensor(weight, dtype=torch.float32).to(device)
            beta_val_tensor = torch.tensor(beta_val, dtype=torch.float32).to(device)

            outputs = model(xs)
            loss = custom_loss(outputs, ys, weights_tensor, ns, beta_val_tensor)
            total_loss += loss.item()  

            loss.backward()  
            optimizer.step()  

    print('Done!')