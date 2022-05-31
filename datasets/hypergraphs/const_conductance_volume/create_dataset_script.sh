# Create constant volume hypergraph
python datasets/hypergraphs/const_conductance_volume/create_hypergraph.py \
 --conductance 0.05 \
 --n 300 \
 --volume 10000 \
 --out_folder datasets/hypergraphs/const_conductance_volume/const_volume/vol_10000

python datasets/hypergraphs/const_conductance_volume/create_hypergraph.py \
 --conductance 0.01 \
 --n 300 \
 --volume 10000 \
 --out_folder datasets/hypergraphs/const_conductance_volume/const_volume/vol_10000

python datasets/hypergraphs/const_conductance_volume/create_hypergraph.py \
 --conductance 0.02 \
 --n 300 \
 --volume 10000 \
 --out_folder datasets/hypergraphs/const_conductance_volume/const_volume/vol_10000

python datasets/hypergraphs/const_conductance_volume/create_hypergraph.py \
 --conductance 0.1 \
 --n 300 \
 --volume 10000 \
 --out_folder datasets/hypergraphs/const_conductance_volume/const_volume/vol_10000


# Create hypergraphs with constant conductance
python datasets/hypergraphs/const_conductance_volume/create_hypergraph.py \
 --conductance 0.05 \
 --n 180 \
 --volume 10000 \
 --out_folder datasets/hypergraphs/const_conductance_volume/const_conductance/cond_0_05

python datasets/hypergraphs/const_conductance_volume/create_hypergraph.py \
 --conductance 0.05 \
 --n 80 \
 --volume 3000 \
 --out_folder datasets/hypergraphs/const_conductance_volume/const_conductance/cond_0_05

python datasets/hypergraphs/const_conductance_volume/create_hypergraph.py \
 --conductance 0.05 \
 --n 120 \
 --volume 5000 \
 --out_folder datasets/hypergraphs/const_conductance_volume/const_conductance/cond_0_05

python datasets/hypergraphs/const_conductance_volume/create_hypergraph.py \
 --conductance 0.05 \
 --n 200 \
 --volume 15000 \
 --out_folder datasets/hypergraphs/const_conductance_volume/const_conductance/cond_0_05