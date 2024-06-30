import torch
import matplotlib.pyplot as plt
import sctour as sct

class SpaRNAVelocity:
    def __init__(self):
        pass
    
    def compute_velocity(self, model, velocity, space):
        """
        Compute the SpaRNA velocity.
        
        Parameters:
        - model (torch.nn.Module): Trained neural network model.
        - velocity (torch.Tensor): Tensor used to compute velocity.
        - space (torch.Tensor): Normalized spatial data.
        
        Returns:
        - predictions_velocity (torch.Tensor): Computed SpaRNA velocity.
        """
        with torch.no_grad():
            velocity = velocity.to('cuda')
            space = space.to('cuda')
            predictions_velocity = model(space+velocity) - model(space)
        predictions_velocity = predictions_velocity.to('cpu')
        return predictions_velocity
    
    def plot_velocity(self, adata, velocity_data):
        """
        Plot the SpaRNA velocity.
        
        Parameters:
        - adata (AnnData): AnnData object containing spatial data.
        - velocity_data (torch.Tensor): Computed SpaRNA velocity.
        """
        adata.obsm['velocity'] = velocity_data.numpy()
        plt.rcParams['figure.figsize'] = (3, 3)
        sct.vf.plot_vector_field(adata, zs_key='spatial', vf_key='velocity', E_key='spatial', 
                                 use_rep_neigh='X_TNODE', color='Layer_Guess', show=False, 
                                 legend_loc='none', frameon=False, size=100, alpha=0.2)
