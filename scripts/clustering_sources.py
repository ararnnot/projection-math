

import numpy as np
import pandas as pd

# Recall: pip install git+https://github.com/ararnnot/projection-math.git
from projection_math.clustering_sources.methods import ClusterSources

def cluster_sources_from_matrix(
    data_projections: np.ndarray,
    source_names: list,
    universe_names: list,
    n_clusters: int,
    distance: str = 'cosine_diss',
    term_fixed: str = None
) -> dict:
    """
    Cluster sources based on their projections.

    Parameters:
        data_projections (np.ndarray): Data matrix.
        source_names (list): Names of the sources.
        universe_names (list): Names of the universes.
        n_clusters (int): Number of clusters.
        distance (str): Distance metric to use for clustering (default: 'cosine_diss').
        term_fixed (str): Fixed term for clustering.

    Returns:
        tuple: Clusters and metrics.
    """
    
    # Create the clustering object
    clusterer = ClusterSources.from_matrix(
        data_projections = data_projections,
        source_names = source_names,
        universe_names = universe_names,
        distance = distance,
        term_fixed = term_fixed
    )
    
    corr = clusterer.compute_correlation()
    clusterer.compute_clustering(n_clusters = n_clusters)
    clusters, clusters_metrics = clusterer.get_clusters()
    clusterer.projection_clusters()
    result_by_cluster = clusterer.get_cluster_projections()
    result_all = clusterer.get_data()

    return {
        "clusters": clusters,
        "metrics": clusters_metrics,
        "result_by_cluster": result_by_cluster,
        "result_all": result_all,
        "corr_matrix": corr
    }
    
def cluster_sources_from_dataframe(
    df: pd.DataFrame,
    n_clusters: int,
    distance: str = 'cosine_diss',
    term_fixed: str = None,
    column_indx: str = 'Source',
    column_indx_features: str = 'Universe',
    column_projection: str = 'Projection',
    column_term: str = 'Term',
) -> dict:
    """
    Cluster sources based on their projections.

    Parameters:
        df (pd.DataFrame): Data.
        source_names (list): Names of the sources.
        universe_names (list): Names of the universes.
        n_clusters (int): Number of clusters.
        distance (str): Distance metric to use for clustering (default: 'cosine_diss').
        term_fixed (str): Fixed term for clustering.

    Returns:
        tuple: Clusters and metrics.
    """
    
    clusterer = ClusterSources.from_dataframe(
        df = df,
        distance = distance,
        term_fixed = term_fixed,
        column_indx = column_indx,
        column_indx_features = column_indx_features,
        column_projection = column_projection,
        column_term = column_term
    )
    
    corr = clusterer.compute_correlation()
    clusterer.compute_clustering(n_clusters = n_clusters)
    clusters, clusters_metrics = clusterer.get_clusters()
    clusterer.projection_clusters()
    result_by_cluster = clusterer.get_cluster_projections()
    result_all = clusterer.get_data()

    return {
        "clusters": clusters,
        "metrics": clusters_metrics,
        "result_by_cluster": result_by_cluster,
        "result_all": result_all,
        "corr_matrix": corr
    }


if __name__ == "__main__":
    
    # Example data with matrix
    data = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.1, 0.4],
        [0.9, 0.8, 0.7]
    ])
    universe_names = ['Mathematics', 'Science', 'Engineering']
    source_names = ['Google', 'DOAJ', 'Arxiv']
    fixed_term = 'Integral'

    results = cluster_sources_from_matrix(
        data_projections = data,
        source_names = source_names,
        universe_names = universe_names,
        n_clusters = 2,
        distance = 'cosine_diss',
        term_fixed = fixed_term
    )

    for key, value in results.items():
        print(f"{key}: \n{value}")
        
        
    # Example data with dataframe
    data_df = pd.DataFrame({
        'Source': ['Google', 'DOAJ', 'Arxiv',
                   'Google', 'DOAJ', 'Arxiv',
                   'Google', 'DOAJ', 'Arxiv'],
        'Universe': ['Mathematics', 'Mathematics', 'Mathematics',
                     'Science', 'Science', 'Science',
                     'Engineering', 'Engineering', 'Engineering'],
        'Projection': [0.1, 0.2, 0.9, 0.2, 0.1, 0.8, 0.3, 0.4, 0.7],
        'Term': ['Integral'] * 9
    })
    fixed_term = 'Integral'
    
    results_df = cluster_sources_from_dataframe(
        df = data_df,
        n_clusters = 2,
        distance = 'cosine_diss',
        term_fixed = fixed_term,
    )
    
    for key, value in results_df.items():
        print(f"{key}: \n{value}")