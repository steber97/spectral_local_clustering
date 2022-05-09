include("methods.jl")
include("methods_hypergraph.jl")

kf = false
ms = 100
mindeg = 5
parts=("sender","recipients")
H = _build_email_hypergraph(data;hyperedgeparts=parts,maxset=ms, keepfauci=kf,mindegree = mindeg)

open("../Hypergraph_clustering_based_on_PageRank/SubmodularHeatEquation/instance/fauci_email.txt", "w") do f
    for (edge, weight) in zip(H.elist, H.weights)
        for node in edge
            # Careful, nodes start from 1, but we want them to start from 0.
            write(f, string(node - 1), " ")
        end
        write(f, string(weight), "\n")
    end
end