import scanpy as sc
import numpy as np
import pandas as pd
import warnings
import anndata as ann
import torch
warnings.filterwarnings("ignore")


def read_data(data_dir, meta_dir, data_type, spatial_coords=('ImageRow', 'ImageCol')):
    """
    Function: Read and combine data and metadata from specified directories into an AnnData object.

    This function reads gene expression data and metadata from specified CSV files.
    It then creates an AnnData object, which is commonly used in single-cell genomics
    analyses. If the data type is spatial transcriptomics (ST), spatial coordinates 
    are also added to the AnnData object.

    Parameters:
    - data_dir (str): The path to the CSV file containing gene expression data. Rows
      should represent genes, and columns should represent samples.
    - meta_dir (str): The path to the CSV file containing metadata. Rows should 
      correspond to samples.
    - data_type (str): The type of data being read. If 'st', spatial coordinates 
      are expected and processed.
    - spatial_coords (tuple of str, optional): The column names in the metadata that
      contain spatial coordinates. Defaults to ('ImageRow', 'ImageCol').

    Output:
    - adata (AnnData): An AnnData object containing the gene expression data, metadata,
      and, if applicable, spatial coordinates.
    """
    
    meta = pd.read_csv(meta_dir, index_col=0)
    data = pd.read_csv(data_dir, index_col=0)
    
    adata = ann.AnnData(data.T)
    adata.obs = meta
    
    if data_type == 'st':
        adata.obsm['spatial'] = np.array(meta[list(spatial_coords)])
    
    return adata



def get_common_hvg(sc_adata, *st_adatas):
    """
    Function: Extracts common highly variable genes (HVGs) from single-cell RNA sequencing (scRNA-seq)
    and spatial transcriptomics (ST) datasets.

    This function identifies HVGs in each dataset using the Seurat v3 method and then 
    finds the intersection of these genes across all datasets. It returns the datasets
    filtered to include only these common HVGs.

    Parameters:
    - sc_adata (AnnData): An AnnData object containing scRNA-seq data.
    - *st_adatas (AnnData): One or more AnnData objects containing ST data.

    Outputs:
    - list of AnnData objects: A list containing the scRNA-seq dataset and all ST datasets,
      each filtered to include only the common HVGs. The first element in the list is the 
      scRNA-seq dataset, followed by the ST datasets.
    """
    
    sc.pp.normalize_total(sc_adata, target_sum=1e4)
    sc.pp.log1p(sc_adata)
    
    sc.pp.highly_variable_genes(sc_adata, flavor="seurat_v3", n_top_genes=3000)
    sc_hvg_set = set(sc_adata.var['highly_variable'][sc_adata.var['highly_variable'] == True].index)

    st_hvg_sets = []
    for st_adata in st_adatas:
        sc.pp.normalize_total(st_adata, target_sum=1e4)
        sc.pp.log1p(st_adata)
        
        sc.pp.highly_variable_genes(st_adata, flavor="seurat_v3", n_top_genes=3000)
        hvg_set = set(st_adata.var['highly_variable'][st_adata.var['highly_variable'] == True].index)
        st_hvg_sets.append(hvg_set)
    
    intersection = sc_hvg_set.intersection(*st_hvg_sets)
    intersection_index = pd.Index(intersection)

    sc_adata = sc_adata[:, list(intersection_index)]
    st_adatas_common_hvg = [st_adata[:, list(intersection_index)] for st_adata in st_adatas]
    
    sorted_hvg = sorted(intersection)
    sc_adata = sc_adata[:, sorted_hvg]
    st_adatas_common_hvg_sorted = [st_adata[:, sorted_hvg] for st_adata in st_adatas_common_hvg]

    return [sc_adata] + st_adatas_common_hvg_sorted

def normalize_spatial_coordinates(adata):
    spatial_data = adata.obsm['spatial']
    spatial_tensor = torch.tensor(spatial_data, dtype=torch.float32)
    
    mean = torch.mean(spatial_tensor, dim=0)
    std = torch.std(spatial_tensor, dim=0)

    spatial_normalized = (spatial_tensor - mean) / (std + 1e-7) 
    
    return spatial_normalized