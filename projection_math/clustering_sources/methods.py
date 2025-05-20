# A Roger Arnau - April 2025

import numpy as np
import pandas as pd
from typing import (
    Union,
    Sequence,
    Any,
    Optional,
)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances, davies_bouldin_score, calinski_harabasz_score
from sklearn_extra.cluster import KMedoids


def _correlation_from_df(df, column_indx_corr, column_indx_features, column_projection):
    """
    Compute the correlation matrix from a DataFrame.
    """
    
    data = df.copy().pivot(
        index = column_indx_corr,
        columns = column_indx_features,
        values = column_projection
    )
    correlation_matrix = pd.DataFrame(
        cosine_similarity(data),
        index = data.index, 
        columns = data.index
    )
    
    return correlation_matrix

def _clustering_from_df(df, n_clusters, distance_matrix,
                       column_indx_corr, column_indx_features, column_projection, index_name = 'Index'):
    """
    Perform clustering on a DataFrame using K-Medoids.
    """
    
    data = df.copy().pivot(
        index = column_indx_corr,
        columns = column_indx_features,
        values = column_projection
    )
    
    distance_matrix[abs(distance_matrix) < 1e-15] = 0
    
    kmedoids = KMedoids(n_clusters=n_clusters, metric="precomputed",
                        method = 'pam', random_state=42)
    clusters = kmedoids.fit_predict(distance_matrix)
        
    cluster_df = pd.DataFrame({index_name: distance_matrix.index, 'Cluster': clusters})
    
    cluster_metrics = {}
    
    cluster_metrics['Mean_Square_Distance'] = {
        cluster: (
            distance_matrix.loc[cluster_df[cluster_df['Cluster'] == cluster][index_name], 
                                cluster_df[cluster_df['Cluster'] == cluster][index_name]]
            .values.flatten() ** 2
        ).mean()
        for cluster in cluster_df['Cluster'].unique()
    }
    
    feature_vectors = data.loc[cluster_df[index_name]].values
    cluster_metrics['Davies_Bouldin_Index'] = davies_bouldin_score(feature_vectors, clusters)
    cluster_metrics['Calinski_Harabasz_Index'] = calinski_harabasz_score(feature_vectors, clusters)
    
    return cluster_df, cluster_metrics


class ClusterSources:
    """
    Class to cluster sources based on their projections.
    
    Attributes:
        distance (str): Distance metric to use for clustering (default: 'cosine_diss').
        term_fixed (str): Fixed term for clustering.
        - Data can be given as a pandas.DataFrame:
        df (pd.DataFrame): DataFrame containing the data.
        column_indx (str): Column name for the index.
        column_indx_features (str): Column name for the features.
        column_projection (str): Column name for the projection.
        column_term (str): Column name for the term.
        - Data can be given as a matrix (source, universe)
        data_projections (np.ndarray): Data matrix.
        source_names (Sequence[str]): Names of the sources.
        universe_names (Sequence[str]): Names of the universes.
    """
    
    def __init__(
        self,
        distance: str = 'cosine_diss',
        term_fixed: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        column_indx: str = 'Source',
        column_indx_features: str = 'Universe',
        column_projection: str = 'Projection',
        column_term: str = 'Term',
        data_projections: Optional[np.ndarray] = None,
        source_names: Optional[Sequence[str]] = None,
        universe_names: Optional[Sequence[str]] = None
    ):
        
        if df is None and data_projections is None:
            raise ValueError("Either df or data_projections must be provided.")
        if df is not None and data_projections is not None:
            raise ValueError("Only one of df or data_projections should be provided.")
        if df is not None:
            self.df = df.copy()
        if data_projections is not None:
            df_matrix = pd.DataFrame(data_projections, columns = universe_names, index = source_names)
            self.df = df_matrix \
                .reset_index() \
                .melt(id_vars = 'index',
                      var_name = column_indx_features,
                      value_name = column_projection) \
                .rename(columns = {'index' : column_indx})
            self.df[column_term] = term_fixed
    
        self.distance           = distance
        self.column_indx        = column_indx
        self.column_indx_features = column_indx_features
        self.column_projection  = column_projection
        self.column_term        = column_term
        self.fix_term(term_fixed)
    
    @classmethod
    def from_matrix(
        cls,
        data_projections: np.ndarray,
        source_names: Sequence[str],
        universe_names: Sequence[str],
        distance: str = 'cosine_diss',
        term_fixed: Optional[str] = None,
    ):
        """
        Create a ClusterSources object from a matrix.
        (see __init__ for details)
        """
        
        return cls(
            data_projections = data_projections,
            source_names = source_names,
            universe_names = universe_names,
            distance = distance,
            term_fixed = term_fixed
        )
        
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        distance: str = 'cosine_diss',
        term_fixed: Optional[str] = None,
        column_indx: str = 'Source',
        column_indx_features: str = 'Universe',
        column_projection: str = 'Projection',
        column_term: str = 'Term'
    ):
        """
        Create a ClusterSources object from a DataFrame.
        (see __init__ for details)
        """
        
        return cls(
            df = df,
            distance = distance,
            term_fixed = term_fixed,
            column_indx = column_indx,
            column_indx_features = column_indx_features,
            column_projection = column_projection,
            column_term = column_term
        )
        
    def __repr__(self):
        return f"Cluster_Sources(df={self.df}, distance={self.distance})"
        
    def fix_term(self, term_fixed):
        """
        Fix the term and report name for the clustering.
        """
        
        self.term_fixed = term_fixed
        if term_fixed is not None:
            self.df = self.df.query(f'{self.column_term} == @term_fixed').drop(columns=[self.column_term])
        
    def check_multiple_terms(self):
        """
        Check if the DataFrame contains multiple terms.
        """
        
        if self.column_term in self.df.columns:
            if len(self.df[self.column_term].unique()) > 1:
                ValueError("Multiple terms found in the DataFrame." \
                    + "Please filter by term before clustering." \
                    + "Use the fix_term method to filter by term.")
        
    def compute_correlation(self):
        """
        Compute the correlation matrix based on the DataFrame.
        
        Returns:
            pd.DataFrame: Correlation matrix.
        """
        
        self.check_multiple_terms()
        
        self.corr = _correlation_from_df(
            self.df,
            column_indx_corr    = self.column_indx,
            column_indx_features = self.column_indx_features,
            column_projection   = self.column_projection,
        )
        return self.corr
    
    def compute_clustering(self, n_clusters):
        """
        Compute clustering based on the correlation matrix.
        
        Returns:
            Clusters
            Cluster metrics (dict)
        """
        
        self.check_multiple_terms()
        
        if self.distance == 'cosine_diss':
            if not hasattr(self, 'corr'):
                print(' (!) Correlation matrix not computed. Computing now...')
                self.compute_correlation()
            self.distance_matrix = 1 - self.corr
            
        elif self.distance in ['euclidean', 'normeuclidean']:
            matrix = self.df.pivot(
                index   = self.column_indx,
                columns = self.column_indx_features,
                values  = self.column_projection
            )
            if self.distance == 'normeuclidean':
                norm = np.linalg.norm(matrix, ord=2, axis=1, keepdims=True)
                norm[norm == 0] = 1
                matrix = matrix.div(norm)  
            self.distance_matrix = euclidean_distances(matrix)
        else:
            raise ValueError(f"Distance {self.distance} not supported. Use 'cosine_diss', 'euclidean', or 'normeuclidean'.")
            
        self.clusters, self.cluster_metrics = _clustering_from_df(
            self.df,
            n_clusters          = n_clusters,
            distance_matrix     = self.distance_matrix,
            column_indx_corr    = self.column_indx,
            column_indx_features = self.column_indx_features,
            column_projection   = self.column_projection,
            index_name          = self.column_indx
        )
        
        return self.clusters, self.cluster_metrics
    
    def projection_clusters(self):
        """
        Join the clusters with the original DataFrame and adds new projections.
        """
        
        self.check_multiple_terms()
                
        self.df = self.df.merge(
            self.clusters,
            on  = self.column_indx,
            how = 'left'
        )
        self.df['Projection_cluster'] = self.df.groupby(['Cluster', self.column_indx_features])['Projection'].transform('mean')
        
    def print_clustering(self):
        """
        Print the clustering results.
        """
        
        print(f'   === Clustering results === ')
        for cluster in self.clusters['Cluster'].unique():
            print(f'Cluster {cluster}: {self.clusters[self.clusters["Cluster"] == cluster][self.column_indx].to_list()}')
        
        print(f'   === Clustering metrics === ')
        for metric, value in self.cluster_metrics.items():
            print(f'{metric}: {value}')    
        
    def get_corr(self):
        if hasattr(self, 'corr'):
            return self.corr
        else:
            raise ValueError("Correlation matrix not computed. Call compute_correlation() first.")
        
    def get_data(self):
        return self.df
    
    def get_clusters(self):
        if hasattr(self, 'clusters'):
            return self.clusters, self.cluster_metrics
        else:
            raise ValueError("Clusters not computed. Call compute_clustering() first.")
    
    def get_cluster_projections(self):
        """
        Return the cluster projections as a pd df like a table.
        """
        return self.df.pivot_table(
            index   = 'Cluster',
            columns = self.column_indx_features,
            values  = 'Projection_cluster'
        )
        
