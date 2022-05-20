include("methods.jl")
include("methods_hypergraph.jl")


# Create dataset without CC and with fauci
kf = true  # Keep fauci
mindeg = 2
parts_no_cc = ("sender","recipients")
H_no_cc = _build_email_hypergraph(data; hyperedgeparts=parts_no_cc, keepfauci=kf,  mindegree = mindeg)

open("../Hypergraph_clustering_based_on_PageRank/SubmodularHeatEquation/instance/fauci_email_no_cc_fauci_LCC.txt", "w") do f
    for (edge, weight) in zip(H_no_cc.elist, H_no_cc.weights)
        for node in edge
            # Careful, nodes start from 1, but we want them to start from 0.
            write(f, string(node - 1), " ")
        end
        write(f, string(weight), "\n")
    end
end


# Create dataset with CC and with fauci
kf = true  # Keep fauci
mindeg = 2
parts_cc = ("sender", "recipients", "cc")
H_cc = _build_email_hypergraph(data; hyperedgeparts=parts_cc, keepfauci=kf,  mindegree = mindeg)

open("../Hypergraph_clustering_based_on_PageRank/SubmodularHeatEquation/instance/fauci_email_cc_fauci_LCC.txt", "w") do f
    for (edge, weight) in zip(H_cc.elist, H_cc.weights)
        for node in edge
            # Careful, nodes start from 1, but we want them to start from 0.
            write(f, string(node - 1), " ")
        end
        write(f, string(weight), "\n")
    end
end

# Create dataset without CC and without fauci
kf = false  # Don't keep fauci
mindeg = 2
parts_no_cc = ("sender","recipients")
H_no_cc = _build_email_hypergraph(data; hyperedgeparts=parts_no_cc, keepfauci=kf,  mindegree = mindeg)

open("../Hypergraph_clustering_based_on_PageRank/SubmodularHeatEquation/instance/fauci_email_no_cc_no_fauci_LCC.txt", "w") do f
    for (edge, weight) in zip(H_no_cc.elist, H_no_cc.weights)
        for node in edge
            # Careful, nodes start from 1, but we want them to start from 0.
            write(f, string(node - 1), " ")
        end
        write(f, string(weight), "\n")
    end
end


# Create dataset with CC and without fauci
kf = false  # Don't keep fauci
mindeg = 2
parts_cc = ("sender", "recipients", "cc")
H_cc = _build_email_hypergraph(data; hyperedgeparts=parts_cc, keepfauci=kf,  mindegree = mindeg)

open("../Hypergraph_clustering_based_on_PageRank/SubmodularHeatEquation/instance/fauci_email_cc_no_fauci_LCC.txt", "w") do f
    for (edge, weight) in zip(H_cc.elist, H_cc.weights)
        for node in edge
            # Careful, nodes start from 1, but we want them to start from 0.
            write(f, string(node - 1), " ")
        end
        write(f, string(weight), "\n")
    end
end