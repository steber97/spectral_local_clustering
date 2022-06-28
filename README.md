# Local Clustering in Hypergraphs (Master Project Stefano Huber 2022 EPFL)

## Installation

Clone with 
```
git clone https://github.com/steber97/spectral_local_clustering.git
```
Move to the downloaded folder, and install the repository dependencies with
```
pip install -r requirements.txt
```
Add the repository to pythonpath:
```
export PYTHONPATH="${PYTHONPATH}:/home/path/to/folder/spectral_local_clustering"
```

## Reproduction of plots

### Figure 1

```
python src/spectral_algo/mixing_in_const_vol_or_conductance_hypergraphs.py --dataset vol_10000
```
Figure is in folder `datasets/hypergraphs/const_conductance_volume/const_conductance/vol_10000/const_vol.png`.

### Figure 2

```
python src/spectral_algo/mixing_in_const_vol_or_conductance_hypergraphs.py --dataset cond_0_05
```
Figure is in folder `datasets/hypergraphs/const_conductance_volume/const_conductance/cond_0_05/const_cond.png`.

### Figure 3

```
python src/spectral_algo/clustering_in_r_uniform_hypergraphs.py --dataset hypergraph_conductance_0_05_vol_10000_n_100
```

Figure is in folder `datasets/hypergraphs/r_uniform/hypergraph_conductance_0_05_vol_10000_n_100/mixing_r_uniform_hypergraph.png`.