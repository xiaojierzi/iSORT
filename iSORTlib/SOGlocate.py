import torch
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_important_genes_and_indices(model, sc_data, num_top_genes, output_dim):
    """
    Function: Calculate the importance scores of each gene in scRNA-seq data using Saliency Maps.

    This function computes the gradient of the output with respect to the input
    (single-cell RNA-seq data) for each sample. These gradients are then averaged
    across samples to determine the importance score for each gene. The function
    returns the indices and names of the top genes based on these importance scores.

    Parameters:
    - model (torch.nn.Module): A trained PyTorch model for scRNA-seq data.
    - sc_adata (Anndata): An Anndata object containing scRNA-seq data where sc_data.X
      is assumed to be a matrix of shape (num_samples, num_features).
    - num_top_genes (int): The number of top genes to be identified based on their
      importance scores.
    - output_dim (int): The dimension of the output layer of the model.
    
    Outputs:
    - top_gene_indices (np.ndarray): Indices of the top genes in the order of
      their importance.
    - top_gene_names (list of str): Names of the top genes corresponding to
      the indices in top_gene_indices.
    """
    gene_names = sc_data.var_names.tolist()
    input_data = sc_data.X

    test = torch.tensor(input_data, dtype=torch.float32, device=device, requires_grad=True)

    output = model(test)

    num_samples, num_features = input_data.shape
    all_grads = np.zeros((num_samples, num_features, output_dim))

    for i in range(num_samples):
        if test.grad is not None:
            test.grad.data.zero_()

        for j in range(output_dim):
            scalar_output = output[i, j]
            scalar_output.backward(retain_graph=True)
            all_grads[i, :, j] = np.abs(test.grad[i].cpu().numpy())

    mean_grads = all_grads.mean(axis=0)

    mean_grads_combined = mean_grads.mean(axis=1)
    sorted_indices_mean = np.argsort(mean_grads_combined)[::-1]

    top_gene_indices = sorted_indices_mean[:num_top_genes]
    top_gene_names = [gene_names[i] for i in top_gene_indices]

    return top_gene_indices, top_gene_names


def knockout_and_visualize(model, test_data, top_genes_indices, sc_obj, color, title):
    """
    Function: Perform gene knockout on test data and visualize the model's predictions.

    This function sets the expression of top genes (specified by indices) to zero
    in the test data, simulating a gene knockout scenario. It then passes this
    modified data through the model to obtain predictions. The predictions are 
    used to create a new embedding in the single-cell object (`sc_obj`). Finally,
    it visualizes this embedding in a 2D plot.
    
    Parameters:
    - model (torch.nn.Module): The trained model to make predictions.
    - test_data (torch.Tensor): The original test data (before gene knockout).
    - top_genes_indices (list or np.ndarray): Indices of the top genes to be 
    knocked out.
    - sc_obj (Anndata): An Anndata object representing the single-cell data. 
    The new embedding will be added to this object.
    - color (str): The name of the variable used to color the points in the plot.
    - title (str): The title of the plot.

    Outputs:
    The function does not return any value but shows a 2D scatter plot of the 
    data points in the new embedding space created post gene knockout. The points 
    are colored based on the 'color' parameter, and the plot is titled according 
    to the 'title' parameter.
    """
    top_genes_indices = np.array(top_genes_indices).copy()
    test_knockout = test_data.clone()
    test_knockout[:, top_genes_indices] = 0
    test_knockout = test_knockout.to(device)
    
    model.eval()
    with torch.no_grad():
        predictions_knockout = model(test_knockout)

    pred_knockout = predictions_knockout.cpu()
    sc_obj.obsm['knockout_space'] = pred_knockout.numpy()
    
    plt.rcParams["figure.figsize"] = (3, 3)
    ax = sc.pl.embedding(sc_obj, basis='knockout_space', color=color, title=title, show=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.show()

