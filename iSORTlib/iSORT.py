import numpy as np
import torch
import warnings
import cvxpy as cp

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KLIEP:
    """
    Function: KLIEP class for fitting scRNA-seq data to different ST (Spatial Transcriptomics) slices.

    This class implements the KLIEP algorithm to adjust the weights of scRNA-seq data
    to better fit ST data slices. It uses a Gaussian kernel for calculating similarities
    and performs an optimization process to adjust the coefficients.

    Parameters:
    - sigma (float): The sigma parameter of the Gaussian kernel, controlling the width of the kernel.
    - step (float): Step length for the optimization process.
    - max_iterations (int): Maximum number of iterations for the optimization process.
    - alpha (array, initially None): Coefficients of basis functions, determined during fitting.
    - X_sc (Tensor, initially None): The gene expression matrix of scRNA-seq data.

    Methods:
    - gaussian_kernel_matrix: Static method to compute the Gaussian kernel matrix.
    - fit: Fits the model to scRNA-seq data, calculating the alpha coefficients.
    - calculate_weights: Calculates weights for each ST sample based on the fitted model.
    """

    def __init__(self, sigma=22, step=0.01, max_iterations=1000):
        self.sigma = sigma
        self.step = step
        self.max_iterations = max_iterations
        self.alpha = None
        self.X_sc = None

    @staticmethod
    def gaussian_kernel_matrix(x1, x2, sigma):
        """
        Function: Computes the Gaussian kernel matrix between two sets of samples.

        Parameters:
        - x1 (Tensor): The first set of samples.
        - x2 (Tensor): The second set of samples.
        - sigma (float): The sigma parameter of the Gaussian kernel.

        Outputs:
        - Tensor: The computed Gaussian kernel matrix.
        """
        dist_sq = torch.cdist(x1, x2, p=2)**2
        return torch.exp(-dist_sq / (2 * sigma ** 2))


    def fit(self, sc_adata):
        """
        Function: Fits the KLIEP model to scRNA-seq data by calculating the alpha coefficients.

        This method normalizes the data using a Gaussian kernel and iteratively adjusts
        the alpha coefficients to optimize the fit to the scRNA-seq data.

        Parameters:
        - sc_adata (Anndata): Anndata object containing scRNA-seq data.
        """

        self.X_sc = torch.tensor(sc_adata.X, dtype=torch.float32).to(device)
        A = self.gaussian_kernel_matrix(self.X_sc, self.X_sc, self.sigma).cpu().numpy()
        b = torch.mean(self.gaussian_kernel_matrix(self.X_sc, self.X_sc, self.sigma), dim=0).cpu().numpy()
        
        n_samples = self.X_sc.shape[0]
        self.alpha = np.random.rand(n_samples)
        
        for _ in range(self.max_iterations):
            self.alpha = self.alpha + self.step * np.dot(A.T, 1. / np.dot(A, self.alpha))
            self.alpha = self.alpha + (1 - np.dot(b.T, self.alpha)) * b / np.dot(b, b)
            self.alpha = np.maximum(0, self.alpha)
            self.alpha = self.alpha / np.dot(b.T, self.alpha)

    def calculate_weights(self, adata_st):
        """
        Function: Calculates weights for each ST sample using the fitted model.

        This method applies the model to a given ST data slice to calculate the importance
        weights of each sample in the slice, based on the learned alpha coefficients.

        Parameters:
        - adata_st (Anndata): Anndata object for a single ST slice.

        Outputs:
        -  Array of calculated weights for each sample in the ST slice.
        """

        X_st = torch.tensor(adata_st.X, dtype=torch.float64).to(device)
        alpha_t = torch.tensor(self.alpha, dtype=torch.float64).to(device)

        def w_gpu(x):
            dist = torch.norm(self.X_sc - x, dim=1)
            kernel_values = torch.exp(-dist ** 2 / (2 * self.sigma ** 2))
            return torch.sum(alpha_t * kernel_values)

        weights = []
        for i in range(X_st.shape[0]):
            weights.append(w_gpu(X_st[i]).cpu().item())
            if i % 1000 == 0:
                print(f"Processed {i} samples out of {X_st.shape[0]}")

        return np.array(weights)
    

def optimize_prediction_weights(sc_adata, *prediction_arrays):
    """
    Function: Estimates weights for each ST slice to optimize predictions.

    This function uses Quadratic Programming to estimate the weights of each ST slice. 
    It constructs a quadratic form using the Laplacian matrix derived from correlations 
    in the scRNA-seq data and minimizes this form subject to constraints.

    Parameters:
    - sc_adata (Anndata): Anndata object containing scRNA-seq data.
    - *prediction_arrays (array): Variable number of arrays containing prediction data for ST slices.

    Outputs:
    - numpy.ndarray: The optimized weights for each ST slice.
    """

    W = np.corrcoef(sc_adata.X.T, rowvar=False)
    D = np.diag(np.sum(W, axis=1))
    L_u = D - W
    H = np.stack(prediction_arrays, axis=-1)

    split_arrays = np.split(H, 2, axis=1)
    H_new = np.vstack([arr.reshape(-1, H.shape[2]) for arr in split_arrays])

    L_u_expanded = np.block([[L_u, np.zeros_like(L_u)],
                             [np.zeros_like(L_u), L_u]])

    Q = H_new.T @ L_u_expanded @ H_new

    beta = cp.Variable(H.shape[2])
    objective = cp.Minimize(cp.quad_form(beta, Q))
    constraints = [cp.sum(beta) == 1, beta >= 0.01]
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return beta.value