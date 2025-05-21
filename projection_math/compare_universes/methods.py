# A Roger Arnau - April 2025

import os
import re
import numpy as np
import pandas as pd
from typing import (
    Sequence,
    Optional,
    Union
)
from gensim.models import KeyedVectors

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances

def preload_word_embedding(embedding_path: str = 'resources/glove.6B.50d.kv'):
    """
    Loads a pretrained word embedding model from the given file path.
    """
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(
            f"The embedding file '{embedding_path}' was not found. Please provide a valid path."
        )
    return KeyedVectors.load(embedding_path)

def _get_term_embedding(term: str, embedding: KeyedVectors) -> np.ndarray:
    # When a term has many words, compute the mean of the vectors
    words = term.split()
    vecs = [embedding[w] for w in words if w in embedding]
    if not vecs:
        try: dim = embedding.vector_size
        except AttributeError: dim = next(iter(embedding.values())).shape[0]
        return np.zeros(dim)
    return np.mean(vecs, axis=0)

def extract_s4_number(s: str) -> int | None:
    """
    Conver to number the extension if its give by its name.
    Extracts the number from a string formatted as "S4X" where X is a digit.
    """
    
    match = re.fullmatch(r"S4(\d+)", s)
    if match:
        return int(match.group(1))
    return None

class Compare_Universes:
    """
    Class to compare two universes based on their word embeddings and projections.
    If a term in the universe is composed by more of one word, the mean of the vectors is used.
    It also admits extension of projection to universe2 if is not given (None),
        in that case, aviod the Error in the extension since is meaningless.
    
    Attributes:
        universe1: Vector of words in the first universe.
        universe2: Vector of words in the second universe.
        projection1: Vector of projection of the first universe.
        projection2: Vector of projection of the second universe.
        model (str): Name of the pre-trained word embedding model.
        distance (str): Distance metric to use ('cosine_diss' or 'euclidean').
    """
    
    def __init__(
        self,
        universe1: Union[Sequence[str], pd.Series, np.ndarray],
        universe2: Union[Sequence[str], pd.Series, np.ndarray],
        projection1: Union[Sequence[float], pd.Series, np.ndarray],
        projection2: Union[Sequence[float], pd.Series, np.ndarray, None] = None,
        word_embedding: Optional[KeyedVectors] = None,
        distance: str = 'cosine_diss'
    ):
        
        if word_embedding is None:
            print(' (!) Word embeddingdoes not exists. Loading now...')
            preload_word_embedding()
        
        if not isinstance(universe1, np.ndarray): universe1 = np.array(universe1)
        if not isinstance(universe2, np.ndarray): universe2 = np.array(universe2)
        if not isinstance(projection1, np.ndarray): projection1 = np.array(projection1)
        if projection2 is None:
            projection2 = np.zeros_like(universe2)
        if not isinstance(projection2, np.ndarray): projection2 = np.array(projection2)
        
        self.universe1 = universe1
        self.universe2 = universe2
        self.projection1 = projection1
        self.projection2 = projection2
        
        self.distance = distance
        self.vec_u1 = np.vstack([
            _get_term_embedding(term, word_embedding)
            for term in universe1 ])
        self.vec_u2 = np.vstack([
            _get_term_embedding(term, word_embedding)
            for term in universe2 ])

        if distance is not None:
            self.compute_distance(distance)

    def __repr__(self):
        return f"Compare_Universes(universe1={self.universe1}, universe2={self.universe2})"

    def compute_distance(self, distance = 'cosine_diss'):
        
        if distance == 'cosine_diss':
            self.d = 1 - cosine_similarity(self.vec_u1, self.vec_u2)
        elif distance == 'euclidean':
            self.d = euclidean_distances(self.vec_u1, self.vec_u2)
        else:
            raise ValueError(f"Distance {distance} not supported. Use 'cosine_diss'.")

        self.Dsum = self.d.sum()
        self.D1 = self.d.sum(axis = 1, keepdims = True)
        self.D2 = self.d.sum(axis = 0, keepdims = True)

    def extension_41(self, ext_universe = 2) -> tuple:
        """
        Extension 4.1: Calculate the extension usign formulas of S4.1
        """

        if ext_universe == 2:
            W2 = ( 1 - self.d / self.D2 ) / ( len(self.universe1) - 1 )
            Ext = self.projection1 @ W2
            Err = (( Ext - self.projection2 )**2).sum()**0.5
            self.E41_Pu2ext = Ext
            self.E41_E2 = Err
        else:
            W1 = ( 1 - self.d.T / self.D1 ) / ( len(self.universe2) - 1 )
            Ext = self.projection2 @ W1
            Err = (( Ext - self.projection1 )**2).sum()**0.5
            self.E41_Pu1ext = Ext
            self.E41_E1 = Err
        
        return Ext, Err
    
    def extension_42(self, ext_universe = 2) -> tuple:
        """
        Extension 4.2: Calculate the extension usign formulas of S4.2
        """
        
        if ext_universe == 2:
            proj = self.projection1
            orig = self.projection2
            dist = self.d
            Ext  = np.zeros_like(self.projection2)
        elif ext_universe == 1:
            proj = self.projection2
            orig = self.projection1
            dist = self.d.T
            Ext  = np.zeros_like(self.projection1)
        
        for i in range(len(Ext)):
            
            act_d = dist[:,i]
            act_d = act_d / act_d.max()
            
            sort_indx = np.argsort(act_d)
            weights = np.zeros_like(act_d)
            
            rest = 1
            if len(sort_indx) > 1:
                for j in sort_indx:
                    if j != sort_indx[-1]:
                        weights[j] = rest * (1 - act_d[j])
                        rest = rest * act_d[j]
                    else:
                        weights[j] = rest
                        
            Ext[i] = np.array(proj) @ weights
        
        Err = (( Ext - orig )**2).sum()**0.5
        
        if ext_universe == 2:
            self.E42_Pu2ext = Ext
            self.E42_E2 = Err
        elif ext_universe == 1:
            self.E42_Pu1ext = Ext
            self.E42_E1 = Err
        
        return Ext, Err
    
    def extension_43(self, conv = 0.5, ext_universe = 2) -> tuple:
        """
        Extension 4.3: Calculate the extension usign formulas of S4.3
        """
        
        if ext_universe == 2:
            proj = self.projection1
            orig = self.projection2
            dist = self.d
            vec  = self.vec_u1
            Ext  = np.zeros_like(self.projection2)
        elif ext_universe == 1:
            proj = self.projection2
            orig = self.projection1
            dist = self.d.T
            vec  = self.vec_u2
            Ext  = np.zeros_like(self.projection1)
            
        if self.distance == 'cosine_diss':
            d_proj = 1 - cosine_similarity(vec, vec)
        elif self.distance == 'euclidean':
            d_proj = euclidean_distances(vec, vec)   
            
        d_proj[d_proj == 0] = 1e-15
        L = ( np.abs( proj - proj.reshape(-1,1) ) / d_proj ).max()
        
        Ext_M = ( proj.reshape(-1,1) - L * dist ).max(axis = 0)
        Ext_W = ( proj.reshape(-1,1) + L * dist ).min(axis = 0)
        Ext   = conv * Ext_M + (1 - conv) * Ext_W
        Err   = (( Ext - orig )**2).sum()**0.5        
        
        if ext_universe == 2:
            self.E43_Pu2ext = Ext
            self.E43_E2 = Err
        elif ext_universe == 1:
            self.E43_Pu1ext = Ext
            self.E43_E1 = Err        
        
        return Ext, Err
        
    def print_extensions(self, ext_universe = 2):
        """
        Print the extensions and errors.
        """
        
        Extensions = np.empty((0, len(self.projection2))) if ext_universe == 2 else np.empty((0, len(self.projection1)))
        Names = np.empty((0, 1), dtype = object)
        
        if ext_universe == 2:
            print(f'   === Real projection onto {self.universe2} === \n'
                  f'{self.projection2} \n')
            if hasattr(self, 'E41_E2'):
                print(f'   === Extension S4.1 === \n'
                    f'{self.E41_Pu2ext} \n'
                    f'Error: {np.round(self.E41_E2, 4)} \n')
            if hasattr(self, 'E42_E2'):
                print(f'   === Extension S4.2 === \n'
                    f'{self.E42_Pu2ext} \n'
                    f'Error: {np.round(self.E42_E2, 4)} \n')
            if hasattr(self, 'E43_E2'):
                print(f'   === Extension S4.3 === \n'
                    f'{self.E43_Pu2ext} \n'
                    f'Error: {np.round(self.E43_E2, 4)} \n')

        elif ext_universe == 1:
            print(f'   === Real projection onto {self.universe1} === \n'
                  f'{self.projection1} \n')
            if hasattr(self, 'E41_E1'):
                print(f'   === Extension S4.1 === \n'
                    f'{self.E41_Pu1ext} \n'
                    f'Error: {np.round(self.E41_E1, 4)} \n')
            if hasattr(self, 'E42_E1'):
                print(f'   === Extension S4.2 === \n'
                    f'{self.E42_Pu1ext} \n'
                    f'Error: {np.round(self.E42_E1, 4)} \n')
            if hasattr(self, 'E43_E1'):
                print(f'   === Extension S4.3 === \n'
                    f'{self.E43_Pu1ext} \n'
                    f'Error: {np.round(self.E43_E1, 4)} \n')    


