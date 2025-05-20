import unittest
import numpy as np
import pandas as pd
from projection_math.clustering_sources.methods import ClusterSources

class TestClusterSources(unittest.TestCase):
    def setUp(self):
        self.data = np.array([
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.4],
            [0.9, 0.8, 0.7]
        ])
        self.universe_names = ['Mathematics', 'Science', 'Engineering']
        self.source_names = ['Google', 'DOAJ', 'Arxiv']
        self.fixed_term = 'Integral'
        self.data_df = pd.DataFrame({
            'Source': ['Google', 'DOAJ', 'Arxiv',
                       'Google', 'DOAJ', 'Arxiv',
                       'Google', 'DOAJ', 'Arxiv'],
            'Universe': ['Mathematics', 'Mathematics', 'Mathematics',
                         'Science', 'Science', 'Science',
                         'Engineering', 'Engineering', 'Engineering'],
            'Projection': [0.1, 0.2, 0.9, 0.2, 0.1, 0.8, 0.3, 0.4, 0.7],
            'Term': ['Integral'] * 9
        })
        
        self.expected_proj_clusters = [0.15, 0.15, 0.90, 0.15, 0.15, 0.80, 0.35, 0.35, 0.70]
        self.expected_clusters = {
            'Source': ['Arxiv', 'DOAJ', 'Google'],
            'Cluster': [0, 1, 1]
        }

    def test_from_matrix(self):
        clusterer = ClusterSources.from_matrix(
            data_projections=self.data,
            source_names=self.source_names,
            universe_names=self.universe_names,
            distance='cosine_diss',
            term_fixed=self.fixed_term
        )
        corr = clusterer.compute_correlation()
        clusterer.compute_clustering(n_clusters=2)
        clusters, metrics = clusterer.get_clusters()
        clusterer.projection_clusters()
    
        self.assertIsNotNone(clusters)
        self.assertIsNotNone(metrics)
        self.assertIsNotNone(corr)

        self.assertListEqual(list(clusters['Source']), self.expected_clusters['Source'])
        self.assertListEqual(list(clusters['Cluster']), self.expected_clusters['Cluster'])

        result_df = clusterer.get_data()
        actual_proj_clusters = list(result_df['Projection_cluster'])
        for actual, expected in zip(actual_proj_clusters, self.expected_proj_clusters):
            self.assertAlmostEqual(actual, expected, places=2)

    def test_from_dataframe(self):
        clusterer = ClusterSources.from_dataframe(
            df=self.data_df,
            distance='cosine_diss',
            term_fixed=self.fixed_term
        )
        corr = clusterer.compute_correlation()
        clusterer.compute_clustering(n_clusters=2)
        clusters, metrics = clusterer.get_clusters()
        clusterer.projection_clusters()
        
        self.assertIsNotNone(clusters)
        self.assertIsNotNone(metrics)
        self.assertIsNotNone(corr)

        self.assertListEqual(list(clusters['Source']), self.expected_clusters['Source'])
        self.assertListEqual(list(clusters['Cluster']), self.expected_clusters['Cluster'])

        result_df = clusterer.get_data()
        actual_proj_clusters = list(result_df['Projection_cluster'])
        for actual, expected in zip(actual_proj_clusters, self.expected_proj_clusters):
            self.assertAlmostEqual(actual, expected, places=2)

if __name__ == '__main__':
    unittest.main()