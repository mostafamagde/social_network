import pandas as pd
import networkx as nx
from community import best_partition, modularity
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt

# Load graph from CSV file
edge_filepath = pd.read_csv("primaryschool_Edges .csv")
node_filepath = pd.read_csv("metadata_primaryschool_Nodes.csv")
#print(edge_filepath)
G = nx.from_pandas_edgelist(edge_filepath,source="Source",target="Target",create_using=nx.MultiGraph())
print("Number of nodes: ", G.number_of_nodes())
print("Number of edges: ", G.number_of_edges())


# Task 1 
#(Louvain algorithm) Find the communities using Louvain algorithm
# Apply Louvain algorithm
partition = best_partition(G)

def visualize_communities(G):
    """Applies the Louvain algorithm and generates a visualization of the graph with
      node colors based on the detected communities."""
    # Generate visualization
    pos = nx.spring_layout(G)
    cmap = plt.cm.tab20
    node_colors = [partition[node] for node in G.nodes()]
    node_sizes = [G.degree(node) for node in G.nodes()]
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, cmap=cmap)
    nx.draw_networkx_edges(G, pos)
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels,font_size=5)
    plt.title('Louvain algorithm')
    plt.colorbar(mappable=plt.cm.ScalarMappable(cmap=cmap), label="Community")
    plt.axis('off')
    plt.show()



# Task 2 At least 3 community detection evaluations ( internal and external evaluation)
# 1- Conductance internal evaluation


def calculate_conductance(G, partition):
    """Calculates the conductance of each community 
    and returns the conductance values for each community."""
    def conductance(G, community):
        Eoc = 0
        Ec = 0
        for node in community:
            neighbors = set(G.neighbors(node))
            for neighbor in neighbors:
                if neighbor not in community:
                    if G.has_edge(node, neighbor):
                        Eoc += G[node][neighbor]['weight'] if G.is_directed() else 1 
                        # it adds the weight of the edge (or 1 if the graph is unweighted) to Eoc.
                else:
                    Ec += G[node][neighbor]['weight'] if G.is_directed() else 1
                    # it adds the weight of the edge (or 1 if the graph is unweighted) to Ec.
        if Ec == 0:
            return 1
        else:
            return 2 * Eoc / (2 * Ec + Eoc)

    communities = {c: [] for c in set(partition.values())}
    for node, community in partition.items():
        communities[community].append(node)

    conductance_values = {f"community {c} : conductance": conductance(G, communities[c]) for c in communities}

    # Return the conductance values for each community
    return conductance_values

# print(calculate_conductance(G, partition))




# 2- Modularity internal evaluation
def calculate_modularity(G):
    """Calculates the modularity of the detected communities and prints the result."""
    communities=list(nx.algorithms.community.greedy_modularity_communities(G))
    modularity=nx.algorithms.community.modularity(G,communities)
    #print(modularity)
    print(f"The modularity of the detected communities is : {modularity:.3f}")


def degree_distribution(G):
# Calculate degrees for all nodes
 degrees = dict(G.degree())

# Calculate degree distribution
 distribution = defaultdict(int)
 for degree in degrees.values():
    distribution[degree] += 1

# Calculate normalized distribution
 total_nodes = len(degrees)
 normalized = {degree: count/total_nodes for degree, count in distribution.items()}
 
 return dict(distribution), dict(normalized)


# 3- Calculate coverage of each community
def calculate_community_coverage(G):
    """Calculates the coverage of each community and prints the result."""
    communities = set(partition.values())
    for community_id in communities:
        community_nodes = [node for node in G.nodes() if partition[node] == community_id]
        internal_edges = G.subgraph(community_nodes).number_of_edges()
        total_edges = sum([G.degree(node) for node in community_nodes])
        coverage = internal_edges / total_edges
        print(f"The coverage of community {community_id} is {coverage:.3f}")
calculate_community_coverage(G)

# 4- Calculate NMI External Evaluation
def calculate_nmi(G, ground_truth_file):
    """Loads the ground truth communities from a CSV file, calculates the NMI between the detected communities
    and the ground truth communities, and prints the result."""
    # Load ground truth communities from CSV file
    ground_truth_dict = dict(zip(ground_truth_file['ID'], ground_truth_file['Class']))
    # Calculate NMI between detected communities and ground truth communities
    nmi = normalized_mutual_info_score(list(ground_truth_dict.values()), list(partition.values()))
    print("NMI: {0:.3f}".format(nmi))
# Task 3 
def calculate_pagerank(G):
    """Calculates the PageRank score for each node in the graph and prints the result."""
    pagerank = nx.pagerank(G)
    for node, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True):
        print(f"Node {node}: PageRank score = {score:.3f}")


def compute_centralities(G):
    """Computes different centrality measures for each node in the graph 
    and returns a DataFrame with the results."""
    G = nx.Graph(G)
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G)
    harmonic_centrality = nx.harmonic_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    degree_counts, degree_probs = degree_distribution(G)


    # Create a DataFrame to store the centrality values for each node
    df = pd.DataFrame(index=G.nodes())
    df.index.name = 'Node ID'
    df['degree'] = pd.Series(dict(G.degree())).astype(int)
    df['degree_centrality'] = pd.Series(degree_centrality).round(3)
    df['betweenness_centrality'] = pd.Series(betweenness_centrality).round(3)
    df['degree_disterbution'] = pd.Series ( degree_probs).round(3)

    df['eigenvector_centrality'] = pd.Series(eigenvector_centrality).round(3)
    df['harmonic_centrality'] = pd.Series(harmonic_centrality).round(3)
    df['closeness_centrality'] = pd.Series(closeness_centrality).round(3)

    df = df.sort_values(by='betweenness_centrality')
    # Export the DataFrame to a CSV file
    df.to_csv('C:/Users/MSI-PC/Desktop/Social Network Task/output.csv')        
    return df







# define the coverage values of the communities
community_coverages = [0.437, 0.424, 0.424, 0.359, 0.449, 0.433]

# calculate the average coverage of the communities
average_coverage = sum(community_coverages) / len(community_coverages)

# print the average coverage
print("The average coverage of the communities is {:.3f}".format(average_coverage))

# x= compute_centralities(G)
# print(x)

# calculate_modularity(G)
# def calculate_conductance(G, partition):
#     """Calculates the conductance of each community 
#     and prints the conductance values for each community."""
#     def conductance(G, community):
#         Eoc = 0
#         Ec = 0
#         for node in community:
#             neighbors = set(G.neighbors(node))
#             for neighbor in neighbors:
#                 if neighbor not in community:
#                     if G.has_edge(node, neighbor):
#                         Eoc += G[node][neighbor]['weight'] if G.is_directed() else 1 
#                         # it adds the weight of the edge (or 1 if the graph is unweighted) to Eoc.
#                 else:
#                     Ec += G[node][neighbor]['weight'] if G.is_directed() else 1
#                     # it adds the weight of the edge (or 1 if the graph is unweighted) to Ec.
#         if Ec == 0:
#             return 1
#         else:
#             return 2 * Eoc / (2 * Ec + Eoc)

#     communities = {c: [] for c in set(partition.values())}
#     for node, community in partition.items():
#         communities[community].append(node)

#     conductance_values = {c: conductance(G, communities[c]) for c in communities}

#     # Print the conductance values for each community
#     for c, value in conductance_values.items():
#         print(f"Community {c}: conductance = {value}")

#     # It is normal for the conductance values to change each time you run the code.
#     #  The Louvain algorithm is a randomized algorithm that uses heuristics to optimize the modularity of the graph partition. 
#     # Therefore, the algorithm may produce slightly different results each time it is run,
#     #  even if the input graph is the same.

# calculate_conductance(G, partition)

def degree_distribution(G):
# Calculate degrees for all nodes
 degrees = dict(G.degree())

# Calculate degree distribution
 distribution = defaultdict(int)
 for degree in degrees.values():
    distribution[degree] += 1

# Calculate normalized distribution
 total_nodes = len(degrees)
 normalized = {degree: count/total_nodes for degree, count in distribution.items()}
 
 return dict(distribution), dict(normalized)
