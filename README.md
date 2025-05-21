
**Projection Math** is an experimental Python package for comparing and analyzing projections across conceptual universes (terms), using word embeddings and mathematical extension techniques. is based on the research papers published as part of the PROMETEO 2024 CIPROM/2023/32 grant.
All code has been developed by Roger Arnau and Enrique Sanchez

---

## Features

- Compare "universes" represented as lists of terms
- Support for pretrained word embeddings (e.g., GloVe)
- Projection extension methods to align semantic spaces
- Custom distance metrics (e.g., cosine dissimilarity)
- Reproducible experiments via scripts in the `scripts/` folder

---

## Installation

pip install git+https://github.com/ararnnot/projection-math.git
from projection_math.clustering_sources.methods import ClusterSources
from projection_math.compare_universes.methods import Compare_Universes
see cripts/ for examples