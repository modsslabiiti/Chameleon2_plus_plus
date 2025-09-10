# ==============================================================================
# Chameleon2++: Ch2++ (PReMI)
# ==============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import re
import time
import math
import heapq
import itertools
import numpy as np
import pandas as pd
import networkx as nx
import metis
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict, deque
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import mutual_info_score
from scipy.stats import entropy
from scipy.optimize import linear_sum_assignment
from annoy import AnnoyIndex

# ==============================================================================
# Logger class for capturing experiment results
# ==============================================================================

class ResultLogger:
    def __init__(self, output_path="Ch2++.csv"):
        """
        Initialize result logger to store experiment metrics.
        
        Parameters:
        - output_path: Path to CSV file for saving results
        """
        self.output_path = output_path
        self.results = []
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(output_path):
            headers = [
                'dataset', '#classes', 'log_type', 'r', 'size (n)', 'knn',
                'knn_type', 'merge_method',
                'm (part)', 'num_nodes', 'num_edges', 'Partitioned (sec)',
                'Partitions after hMETIS',
                'Partitions after Flood-Fill', 'alpha', 'beta', 'Number of final clusters after merging',
                'NMI-Score', 'Max NMI Score', 'Max NMI at Clusters', 
                'Accuracy', 'Max Accuracy Score', 'Max Accuracy at Clusters',
                'Adjusted Rand Index', 'Total execution time (sec)'
            ]
            df = pd.DataFrame(columns=headers)
            df.to_csv(output_path, index=False)
    
    def log_result(self, result_dict):
        """
        Log a single experiment result.
        
        Parameters:
        - result_dict: Dictionary containing experiment metrics
        """
        self.results.append(result_dict)
        
        # Append to CSV file
        df = pd.DataFrame([result_dict])
        df.to_csv(self.output_path, mode='a', header=False, index=False)
    
    def print_table(self, result_dict):
        """
        Print a formatted table of the latest experiment result.
        
        Parameters:
        - result_dict: Dictionary containing experiment metrics
        """
        print("\n" + "="*60)
        print("Experiment Results Summary")
        print("="*60)
        
        # Calculate padding for formatting
        pad_size = max([len(key) for key in result_dict.keys()]) + 2
        
        # Print each metric
        for key, value in result_dict.items():
            print(f"{key.ljust(pad_size)}: {value}")
        
        print("="*60 + "\n")

# ==============================================================================
# K-NN GRAPH CONSTRUCTION FUNCTIONS
# ==============================================================================

def build_annoy_index(points, n_trees, verbose=False):
    """Build and return Annoy index for the given points"""
    if verbose:
        print("Building Annoy index...")
    
    n_dims = len(points[0])
    annoy_index = AnnoyIndex(n_dims, 'euclidean')
    
    for i, point in enumerate(points):
        annoy_index.add_item(i, point)
    
    annoy_index.build(n_trees)
    
    if verbose:
        print(f"Annoy index built with {len(points)} points and {n_trees} trees")
    
    return annoy_index

def knn_graph_asymmetric_annoy(df, knn, n_trees, verbose=False):
    """Build asymmetric k-NN graph using Annoy"""
    points = np.array([p[1:] for p in df.itertuples()])
    
    if verbose:
        print(f"Building asymmetric k-NN with Annoy (knn = {knn}, n_trees = {n_trees})...")
    
    # Build Annoy index
    annoy_index = build_annoy_index(points, n_trees, verbose)
    
    g = nx.Graph()
    g.add_nodes_from(range(len(points)))
    
    iterator = tqdm(range(len(points)), total=len(points)) if verbose else range(len(points))
    for i in iterator:
        # Get k+1 nearest neighbors (including self)
        neighbors = annoy_index.get_nns_by_item(i, knn + 1)[1:]  # exclude self
        
        for neighbor_idx in neighbors:
            distance = annoy_index.get_distance(i, neighbor_idx)
            if distance > 0:
                weight = max(1, min(1000, int(1000.0 / (distance + 1e-10))))
                g.add_edge(i, neighbor_idx, weight=weight)
        
        g.nodes[i]['pos'] = points[i]
    
    g.graph['edge_weight_attr'] = 'weight'
    
    if verbose:
        print(f"Asymmetric Annoy k-NN graph created with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
        if g.number_of_edges() > 0:
            print(f"Weight range: {min(d['weight'] for u, v, d in g.edges(data=True))} to {max(d['weight'] for u, v, d in g.edges(data=True))}")
    
    return g

def knn_graph_symmetric_annoy(df, knn, n_trees, verbose=False):
    """Build symmetric k-NN graph using Annoy"""
    points = np.array([p[1:] for p in df.itertuples()])
    
    if verbose:
        print(f"Building symmetric k-NN with Annoy (knn = {knn}, n_trees = {n_trees})...")
    
    # Build Annoy index
    annoy_index = build_annoy_index(points, n_trees, verbose)
    
    # Build k-NN sets for each node for symmetric check
    knn_sets = {}
    for i in range(len(points)):
        neighbors = annoy_index.get_nns_by_item(i, knn + 1)[1:]  # exclude self
        knn_sets[i] = set(neighbors)
    
    g = nx.Graph()
    g.add_nodes_from(range(len(points)))
    
    iterator = tqdm(range(len(points)), total=len(points)) if verbose else range(len(points))
    for i in iterator:
        neighbors = annoy_index.get_nns_by_item(i, knn + 1)[1:]  # exclude self
        
        for neighbor_idx in neighbors:
            # Only add edge if mutual k-NN (symmetric condition)
            if i in knn_sets[neighbor_idx]:
                distance = annoy_index.get_distance(i, neighbor_idx)
                if distance > 0:
                    weight = max(1, min(1000, int(1000.0 / (distance + 1e-10))))
                    g.add_edge(i, neighbor_idx, weight=weight)
        
        g.nodes[i]['pos'] = points[i]
    
    g.graph['edge_weight_attr'] = 'weight'
    
    if verbose:
        print(f"Symmetric Annoy k-NN graph created with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
        if g.number_of_edges() > 0:
            print(f"Weight range: {min(d['weight'] for u, v, d in g.edges(data=True))} to {max(d['weight'] for u, v, d in g.edges(data=True))}")
    
    return g

def build_knn_graph_annoy(df, knn, knn_type, n_trees, verbose=False):
    """
    Build k-NN graph using Annoy with specified type.
    
    Parameters:
    - df: DataFrame with features
    - knn: Number of nearest neighbors
    - knn_type: 'symmetric' or 'asymmetric'
    - n_trees: Number of trees for Annoy index (more trees = better accuracy, slower build)
    - verbose: Whether to display progress
    
    Returns:
    - NetworkX graph
    """
    start_time = time.time()
    
    if knn_type == 'symmetric':
        graph = knn_graph_symmetric_annoy(df, knn, n_trees, verbose)
    elif knn_type == 'asymmetric':
        graph = knn_graph_asymmetric_annoy(df, knn, n_trees, verbose)
    else:
        raise ValueError(f"Unknown knn_type: {knn_type}. Must be 'symmetric' or 'asymmetric'")
    
    if verbose:
        print(f"Total graph construction time: {time.time() - start_time:.2f} seconds")
        print("- - - - - -")
    
    # Return sorted version of the graph
    H = nx.Graph()
    H.add_nodes_from(sorted(graph.nodes(data=True)))
    H.add_edges_from(graph.edges(data=True))
    H.graph.update(graph.graph)  
    
    return H

# ==============================================================================
# PARTITIONING AND REFINEMENT FUNCTIONS 
# ==============================================================================

def optimized_pre_part_graph(graph, k, df=None, verbose=False):
    """
    Optimized initial partitioning using hMETIS.
    """
    partition_start = time.time()
    
    if verbose:
        print("Begin Partitioning...")
    
    if k <= 0:
        raise ValueError("Number of partitions k must be positive")
    if k >= graph.number_of_nodes():
        if verbose:
            print(f"Warning: k={k} >= number of nodes. Each node will be its own partition.")
    
    try:
        edgecuts, parts = metis.part_graph(graph, k, recursive=False)
    except Exception as e:
        if verbose:
            print(f"METIS partitioning failed: {e}")
        parts = list(range(len(graph.nodes()))) % k
    
    partition_time = time.time() - partition_start
    
    if verbose:
        print(f"Partitioned in {partition_time:.2f} seconds")
        print(f"Number of unique partitions: {len(set(parts))}")
        print(f"Partitioned graph into {k} parts with {edgecuts} edge cuts")
    
    cluster_attr = {i: part for i, part in enumerate(parts)}
    nx.set_node_attributes(graph, cluster_attr, 'cluster')
    
    if df is not None:
        df['cluster'] = [cluster_attr[i] for i in range(len(df))]
    
    return graph, parts, partition_time

def optimized_flood_fill_refinement(graph, initial_partitions, verbose=False):
    """
    Optimized flood fill refinement using deque.
    """
    if verbose:
        print("Refining partitions with optimized flood fill...")
        start = time.time()
    
    partition_groups = defaultdict(list)
    for node, partition in enumerate(initial_partitions):
        partition_groups[partition].append(node)
    
    adj_list = {node: set(graph.neighbors(node)) for node in graph.nodes()}
    refined_partitions = []
    
    for partition_id, nodes in partition_groups.items():
        if not nodes:
            continue
        unvisited = set(nodes)
        while unvisited:
            start_node = unvisited.pop()
            component = []
            queue = deque([start_node])
            visited_in_component = {start_node}
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in adj_list[current]:
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        visited_in_component.add(neighbor)
                        queue.append(neighbor)
            refined_partitions.append(component)
    
    if verbose:
        print(f"Refinement completed in {time.time() - start:.2f} seconds")
        print(f"Original partitions: {len(partition_groups)}")
        print(f"Refined partitions: {len(refined_partitions)}")
    
    return refined_partitions

def update_graph_and_dataframe(graph, df, refined_partitions, verbose=False):
    """
    Update graph and dataframe with refined partition assignments.
    """
    if verbose:
        print("Updating cluster assignments...")
    
    cluster_mapping = {}
    for cluster_id, nodes in enumerate(refined_partitions):
        for node in nodes:
            cluster_mapping[node] = cluster_id
    
    nx.set_node_attributes(graph, cluster_mapping, 'cluster')
    
    if df is not None:
        df['cluster'] = [cluster_mapping[i] for i in range(len(df))]
    
    return len(refined_partitions)

def complete_optimized_partitioning(graph, k, df=None, verbose=False):
    """
    Complete optimized partitioning pipeline.
    """
    total_start = time.time()
    
    graph, initial_parts, partition_time = optimized_pre_part_graph(graph, k, df=None, verbose=verbose)
    refined_partitions = optimized_flood_fill_refinement(graph, initial_parts, verbose=verbose)
    final_partition_count = update_graph_and_dataframe(graph, df, refined_partitions, verbose=verbose)
    
    if verbose:
        print(f"Total partitioning time: {time.time() - total_start:.2f} seconds")
        print(f"Final number of partitions: {final_partition_count}")
    
    return graph, refined_partitions, partition_time, len(set(initial_parts))
    
def hungarian_accuracy(true_labels, pred_labels):
    # Compute the confusion matrix
    #num_classes = np.unique(true_labels).shape[0]
    #confusion_matrix = np.zeros((num_classes, num_classes))
    #for i in range(num_classes):
    #    for j in range(num_classes):
    #        confusion_matrix[i, j] = np.sum(np.logical_and(true_labels == i, pred_labels == j))
    
    # Apply the Hungarian algorithm
    #row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    #accuracy = confusion_matrix[row_ind, col_ind].sum() / confusion_matrix.sum()
    
    #return accuracy
    
    """
    Calculate accuracy using Hungarian algorithm for optimal label assignment.
    Handles arbitrary label values by mapping them to consecutive integers.
    """
    # Convert to numpy arrays if not already
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    # Get unique labels and map them to consecutive integers starting from 0
    true_unique = np.unique(true_labels)
    pred_unique = np.unique(pred_labels)
    
    # Create mapping dictionaries
    true_mapping = {label: i for i, label in enumerate(true_unique)}
    pred_mapping = {label: i for i, label in enumerate(pred_unique)}
    
    # Map labels to consecutive integers
    true_mapped = np.array([true_mapping[label] for label in true_labels])
    pred_mapped = np.array([pred_mapping[label] for label in pred_labels])
    
    # Get the number of classes (use the maximum to handle cases where one has more classes)
    num_true_classes = len(true_unique)
    num_pred_classes = len(pred_unique)
    max_classes = max(num_true_classes, num_pred_classes)
    
    # Create confusion matrix
    confusion_matrix = np.zeros((max_classes, max_classes))
    
    for i in range(len(true_mapped)):
        confusion_matrix[true_mapped[i], pred_mapped[i]] += 1
    
    # Apply the Hungarian algorithm (we use negative because linear_sum_assignment finds minimum)
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    
    # Compute the accuracy
    total_correct = confusion_matrix[row_ind, col_ind].sum()
    total_samples = len(true_labels)
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    return accuracy

# ==============================================================================
# NAIVE MERGING FUNCTIONS
# ==============================================================================

def get_cluster_nodes(graph, clusters):
    """Get nodes belonging to specified clusters"""
    return [n for n in graph.nodes if graph.nodes[n]["cluster"] in clusters]

def get_connecting_edges(partitions, graph):
    """Find edges connecting two partitions"""
    edges = []
    for a in partitions[0]:
        for b in partitions[1]:
            if a in graph and b in graph[a]:
                edges.append((a, b))
    return edges

def get_edge_weights(graph, edges):
    """Get weights of specified edges"""
    return [graph[edge[0]][edge[1]]["weight"] for edge in edges]

def internal_edge_sum(graph, cluster):
    """Sum of weights of internal edges in a cluster"""
    subgraph = graph.subgraph(cluster)
    edges = subgraph.edges()
    weights = get_edge_weights(subgraph, edges)
    return np.sum(weights)

def internal_edge_count(graph, cluster):
    """Number of internal edges in a cluster"""
    subgraph = graph.subgraph(cluster)
    return len(subgraph.edges())

def average_internal_weight(graph, cluster):
    """Average weight of internal edges in a cluster"""
    weight_sum = internal_edge_sum(graph, cluster)
    edge_count = internal_edge_count(graph, cluster)
    return weight_sum / edge_count if edge_count > 0 else 0

def average_connecting_weight(graph, cluster_i, cluster_j):
    """Average weight of edges connecting two clusters"""
    edges = get_connecting_edges((cluster_i, cluster_j), graph)
    weights = get_edge_weights(graph, edges)
    return np.mean(weights) if len(weights) > 0 else 0

def connecting_edge_count(graph, cluster_i, cluster_j):
    """Number of edges connecting two clusters"""
    edges = get_connecting_edges((cluster_i, cluster_j), graph)
    return len(edges)

def proximity_ratio(graph, cluster_i, cluster_j):
    """Proximity ratio between two clusters (min/max of average internal weights)"""
    avg_i = average_internal_weight(graph, cluster_i)
    avg_j = average_internal_weight(graph, cluster_j)
    if avg_i == 0 or avg_j == 0:
        return 0
    return min(avg_i, avg_j) / max(avg_i, avg_j)

def relative_closeness(graph, cluster_i, cluster_j):
    """
    Calculate relative closeness between clusters
    """
    edge_count_i = internal_edge_count(graph, cluster_i)
    edge_count_j = internal_edge_count(graph, cluster_j)
    weight_sum_i = internal_edge_sum(graph, cluster_i)
    weight_sum_j = internal_edge_sum(graph, cluster_j)
    
    avg_connecting_weight = average_connecting_weight(graph, cluster_i, cluster_j)
    
    # Handle edge cases
    if weight_sum_i + weight_sum_j == 0:
        return 0
        
    common_factor = avg_connecting_weight / (weight_sum_i + weight_sum_j)
    
    # Special case for singleton clusters
    if edge_count_i == 0 or edge_count_j == 0:
        return 1e3 * common_factor
    else:
        return (edge_count_i + edge_count_j) * common_factor

def relative_interconnectivity(graph, cluster_i, cluster_j, beta):
    """
    Calculate relative interconnectivity between clusters
    """
    edge_count_i = internal_edge_count(graph, cluster_i)
    edge_count_j = internal_edge_count(graph, cluster_j)
    
    edge_count_ij = connecting_edge_count(graph, cluster_i, cluster_j)
    prox_ratio = proximity_ratio(graph, cluster_i, cluster_j)
    
    # Handle edge cases
    if edge_count_i == 0 or edge_count_j == 0:
        return 1  # Allow merging isolated clusters
    
    if min(edge_count_i, edge_count_j) == 0:
        return 0
        
    return (edge_count_ij / min(edge_count_i, edge_count_j)) * (prox_ratio ** beta)

def calculate_merge_score(graph, cluster_i, cluster_j, alpha, beta):
    """Calculate the final merge score combining RI and RC"""
    ri = relative_interconnectivity(graph, cluster_i, cluster_j, beta)
    rc = relative_closeness(graph, cluster_i, cluster_j)
    return ri * (rc ** alpha)

def find_and_merge_best_clusters_naive(graph, df, alpha, beta, target_clusters, verbose=False):
    """
    Find the best pair of clusters to merge based on merge score (naive approach).
    
    Parameters:
    - graph: NetworkX graph with cluster assignments
    - df: DataFrame with cluster assignments
    - alpha: Parameter controlling the influence of relative closeness
    - beta: Parameter controlling the influence of proximity ratio
    - target_clusters: Target number of clusters to reach
    - verbose: Whether to display detailed information
    
    Returns:
    - Boolean indicating if a merge was performed
    """
    # Get unique cluster IDs
    clusters = np.unique(df["cluster"])
    
    # If already at or below target, no need to merge
    if len(clusters) <= target_clusters:
        return False

    # Find best pair to merge
    max_score = 0
    best_pair = (-1, -1)
    
    # Evaluate all pairs of clusters
    for i in clusters:
        for j in clusters:
            if i >= j:  # Skip self-comparisons and duplicates
                continue
                
            # Get nodes in each cluster
            nodes_i = get_cluster_nodes(graph, [i])
            nodes_j = get_cluster_nodes(graph, [j])
            
            # Check if there are any connecting edges
            edges = get_connecting_edges((nodes_i, nodes_j), graph)
            if not edges:
                continue
                
            # Calculate merge score
            merge_score = calculate_merge_score(graph, nodes_i, nodes_j, alpha, beta)
                
            # Update best pair if score is higher
            if merge_score > max_score:
                max_score = merge_score
                best_pair = (i, j)

    # Perform the merge if a valid pair was found
    if max_score > 0:
        ci, cj = best_pair
            
        # Update cluster assignments in DataFrame
        df.loc[df["cluster"] == cj, "cluster"] = ci
        
        # Update cluster assignments in graph
        for node in graph.nodes():
            if graph.nodes[node]["cluster"] == cj:
                graph.nodes[node]["cluster"] = ci
                
        return True
    
    return False

#=====================================================================================
# Merging with NMI tracking and complete merging: 
# NOTE: For optimal merging (run-time), erfer the merging code of INDICON (Approx-Ch)
#=====================================================================================

class OptimizedGraphClusterer:
    """
    Optimized graph clustering class with priority queue-based merging and NMI tracking.
    """
    def __init__(self, graph, df, alpha, beta, true_labels=None):
        self.graph = graph
        self.df = df
        self.alpha = alpha
        self.beta = beta
        self.true_labels = true_labels
        self.adjacency_cache = {}
        self.score_cache = {}        
        self.nmi_history = []  # Track NMI at each merge step
        self.accuracy_history = []  # Track accuracy at each merge step
        self._build_cluster_adjacency()
    
    def _build_cluster_adjacency(self):
        """Build adjacency matrix for clusters to avoid repeated edge computations"""
        clusters = np.unique(self.df['cluster'])
        self.cluster_adjacency = defaultdict(set)
        
        # Build cluster-to-cluster adjacency
        for edge in self.graph.edges():
            node1, node2 = edge
            cluster1 = self.graph.nodes[node1]['cluster']
            cluster2 = self.graph.nodes[node2]['cluster']
            
            if cluster1 != cluster2:
                self.cluster_adjacency[cluster1].add(cluster2)
                self.cluster_adjacency[cluster2].add(cluster1)
                
    def calculate_current_accuracy(self):
        """Calculate accuracy for current clustering state"""
        if self.true_labels is None:
            return 0.0
        
        normalized_df = normalize_cluster_labels(self.df)
        predicted_labels = normalized_df["cluster"].to_numpy()
        return hungarian_accuracy(self.true_labels, predicted_labels)
    
    def get_cluster(self, clusters):
        """Get nodes belonging to specified clusters"""
        if isinstance(clusters, (int, np.integer)):
            clusters = [clusters]
        return [n for n in self.graph.nodes if self.graph.nodes[n]['cluster'] in clusters]
    
    def connecting_edges(self, cluster_i_nodes, cluster_j_nodes):
        """Get edges connecting two clusters"""
        cut_set = []
        # Convert to sets for faster lookup
        cluster_j_set = set(cluster_j_nodes)
        
        for node_a in cluster_i_nodes:
            if node_a in self.graph:
                neighbors = set(self.graph[node_a].keys())
                connecting_neighbors = neighbors.intersection(cluster_j_set)
                for node_b in connecting_neighbors:
                    cut_set.append((node_a, node_b))
        return cut_set
    
    def get_weights(self, edges):
        """Get weights for given edges"""
        return [self.graph[edge[0]][edge[1]]['weight'] for edge in edges]
    
    def s_ci(self, cluster_nodes):
        """Sum of weights within a cluster"""
        cluster_subgraph = self.graph.subgraph(cluster_nodes)
        edges = cluster_subgraph.edges()
        weights = self.get_weights(edges)
        return np.sum(weights) if weights else 0
    
    def E_ci(self, cluster_nodes):
        """Number of edges within a cluster"""
        cluster_subgraph = self.graph.subgraph(cluster_nodes)
        return len(cluster_subgraph.edges())
    
    def s_ci_avg(self, cluster_nodes):
        """Average weight within a cluster"""
        sci = self.s_ci(cluster_nodes)
        eci = self.E_ci(cluster_nodes)
        return sci / eci if eci > 0 else 0
    
    def s_cij_avg(self, cluster_i_nodes, cluster_j_nodes):
        """Average weight between two clusters"""
        edges = self.connecting_edges(cluster_i_nodes, cluster_j_nodes)
        if not edges:
            return 0
        weights = self.get_weights(edges)
        return np.mean(weights)
    
    def E_cij(self, cluster_i_nodes, cluster_j_nodes):
        """Number of edges between two clusters"""
        edges = self.connecting_edges(cluster_i_nodes, cluster_j_nodes)
        return len(edges)
    
    def p_cij(self, cluster_i_nodes, cluster_j_nodes):
        """Proportion measure between clusters"""
        scia = self.s_ci_avg(cluster_i_nodes)
        scja = self.s_ci_avg(cluster_j_nodes)
        
        if scia == 0 and scja == 0:
            return 1
        if scia == 0 or scja == 0:
            return 0
        
        min_sci_scj_a = min(scia, scja)
        max_sci_scj_a = max(scia, scja)
        return min_sci_scj_a / max_sci_scj_a
    
    def rc(self, cluster_i_nodes, cluster_j_nodes):
        """Relative connectivity measure"""
        eci = self.E_ci(cluster_i_nodes)
        ecj = self.E_ci(cluster_j_nodes)
        sci = self.s_ci(cluster_i_nodes)
        scj = self.s_ci(cluster_j_nodes)
        scija = self.s_cij_avg(cluster_i_nodes, cluster_j_nodes)
        
        if sci + scj == 0:
            return 0
        
        common_fact = scija / (sci + scj)
        
        if eci == 0 or ecj == 0:
            return 1e3 * common_fact
        else:
            return (eci + ecj) * common_fact
    
    def ri(self, cluster_i_nodes, cluster_j_nodes):
        """Relative interconnectivity measure"""
        eci = self.E_ci(cluster_i_nodes)
        ecj = self.E_ci(cluster_j_nodes)
        ecij = self.E_cij(cluster_i_nodes, cluster_j_nodes)
        pcij = self.p_cij(cluster_i_nodes, cluster_j_nodes)
        
        if eci == 0 or ecj == 0:
            return 1
        
        min_edges = min(eci, ecj)
        if min_edges == 0:
            return 1
        
        return (ecij / min_edges) * np.power(pcij, self.beta)
    
    def merge_score(self, cluster_i_nodes, cluster_j_nodes):
        """Calculate merge score between two clusters"""
        ri_val = self.ri(cluster_i_nodes, cluster_j_nodes)
        rc_val = self.rc(cluster_i_nodes, cluster_j_nodes)
        return ri_val * np.power(rc_val, self.alpha)
    
    def calculate_current_nmi(self):
        """Calculate NMI for current clustering state"""
        if self.true_labels is None:
            return 0.0
        
        normalized_df = normalize_cluster_labels(self.df)
        predicted_labels = normalized_df["cluster"].to_numpy()
        return normalized_mutual_info_score(self.true_labels, predicted_labels)
    
    def initialize_priority_queue(self):
        """Initialize priority queue with all valid cluster pairs"""
        clusters = np.unique(self.df['cluster'])
        priority_queue = []
        
        for i, j in itertools.combinations(clusters, 2):
            if i in self.cluster_adjacency and j in self.cluster_adjacency[i]:
                gi = self.get_cluster(i)
                gj = self.get_cluster(j)
                
                # Check if clusters are connected
                edges = self.connecting_edges(gi, gj)
                if edges:
                    score = self.merge_score(gi, gj)
                    # Use negative score for max heap behavior
                    heapq.heappush(priority_queue, (-score, i, j))
        
        return priority_queue
    
    def find_disconnected_components(self):
        """Find disconnected components in current clustering"""
        clusters = np.unique(self.df['cluster'])
        cluster_graph = nx.Graph()
        cluster_graph.add_nodes_from(clusters)
        
        # Add edges between connected clusters
        for cluster_i in clusters:
            if cluster_i in self.cluster_adjacency:
                for cluster_j in self.cluster_adjacency[cluster_i]:
                    if cluster_j in clusters:
                        cluster_graph.add_edge(cluster_i, cluster_j)
        
        # Find connected components
        components = list(nx.connected_components(cluster_graph))
        return components
    
    def merge_within_components(self, components, verbose=False):
        """Merge clusters within each disconnected component"""
        total_merges = 0
        
        for component in components:
            if len(component) <= 1:
                continue
                
            # Create priority queue for this component only
            component_queue = []
            
            for i, j in itertools.combinations(component, 2):
                if i in self.cluster_adjacency and j in self.cluster_adjacency[i]:
                    gi = self.get_cluster(i)
                    gj = self.get_cluster(j)
                    
                    edges = self.connecting_edges(gi, gj)
                    if edges:
                        score = self.merge_score(gi, gj)
                        heapq.heappush(component_queue, (-score, i, j))
            
            # Merge within this component until only one cluster remains
            component_clusters = set(component)
            while len(component_clusters) > 1 and component_queue:
                neg_score, ci, cj = heapq.heappop(component_queue)
                
                if ci not in component_clusters or cj not in component_clusters:
                    continue
                
                # Perform the merge
                self.merge_clusters(ci, cj)
                component_clusters.discard(cj)
                total_merges += 1
                
                # Calculate and store NMI and accuracy
                current_nmi = self.calculate_current_nmi()
                current_accuracy = self.calculate_current_accuracy()
                current_cluster_count = len(np.unique(self.df['cluster']))
                self.nmi_history.append((current_cluster_count, current_nmi))
                self.accuracy_history.append((current_cluster_count, current_accuracy))
                
                if verbose:
                    print(f"Merged clusters {cj} -> {ci}, Clusters: {current_cluster_count}, NMI: {current_nmi:.4f}, Accuracy: {current_accuracy:.4f}")
        
        return total_merges
    
    def update_priority_queue_after_merge(self, priority_queue, merged_cluster, removed_cluster):
        """Update priority queue after a merge operation"""
        # Add new entries for the merged cluster
        clusters = np.unique(self.df['cluster'])
        
        for other_cluster in clusters:
            if (other_cluster != merged_cluster and
                other_cluster in self.cluster_adjacency and
                merged_cluster in self.cluster_adjacency[other_cluster]):
                
                gi = self.get_cluster(merged_cluster)
                gj = self.get_cluster(other_cluster)
                
                edges = self.connecting_edges(gi, gj)
                if edges:
                    score = self.merge_score(gi, gj)
                    heapq.heappush(priority_queue, (-score, merged_cluster, other_cluster))
    
    def is_valid_pair(self, cluster_i, cluster_j):
        """Check if a cluster pair is still valid"""
        current_clusters = set(np.unique(self.df['cluster']))
        return cluster_i in current_clusters and cluster_j in current_clusters
    
    def merge_clusters(self, ci, cj):
        """Merge cluster cj into cluster ci"""
        # Update dataframe
        self.df.loc[self.df['cluster'] == cj, 'cluster'] = ci
        
        # Update graph nodes
        for node in self.graph.nodes():
            if self.graph.nodes[node]['cluster'] == cj:
                self.graph.nodes[node]['cluster'] = ci
        
        # Update cluster adjacency
        if cj in self.cluster_adjacency:
            # Create a copy of the neighbors to avoid "set changed size during iteration"
            cj_neighbors = set(self.cluster_adjacency[cj])
            
            # Merge adjacencies
            self.cluster_adjacency[ci].update(cj_neighbors)
            self.cluster_adjacency[ci].discard(ci)  # Remove self-reference
            self.cluster_adjacency[ci].discard(cj)  # Remove reference to merged cluster
            
            # Update other clusters' references
            for neighbor in cj_neighbors:
                if neighbor in self.cluster_adjacency:
                    self.cluster_adjacency[neighbor].discard(cj)
                    self.cluster_adjacency[neighbor].add(ci)
            
            # Remove the merged cluster
            del self.cluster_adjacency[cj]
    
    def complete_merge_to_one(self, verbose=False):
        """Merge all clusters down to 1, handling disconnected components"""
        # Initialize NMI tracking
        initial_nmi = self.calculate_current_nmi()
        initial_accuracy = self.calculate_current_accuracy()
        initial_clusters = len(np.unique(self.df['cluster']))
        self.nmi_history = [(initial_clusters, initial_nmi)]
        self.accuracy_history = [(initial_clusters, initial_accuracy)]
        
        if verbose:
            print(f"Starting complete merge from {initial_clusters} clusters to 1")
            print(f"Initial NMI: {initial_nmi:.4f}, Initial Accuracy: {initial_accuracy:.4f}")
        
        priority_queue = self.initialize_priority_queue()
        current_cluster_count = len(np.unique(self.df['cluster']))
        
        successful_merges = 0
        iterations = 0
        max_iterations = current_cluster_count * 10  # Prevent infinite loops
        
        pbar = tqdm(total=current_cluster_count-1, desc="Merging to 1 cluster")
        
        # Phase 1: Merge connected clusters
        while (current_cluster_count > 1 and 
               priority_queue and 
               iterations < max_iterations):
            
            iterations += 1
            
            # Get the best scoring pair
            neg_score, ci, cj = heapq.heappop(priority_queue)
            score = -neg_score
            
            # Check if this pair is still valid
            if not self.is_valid_pair(ci, cj):
                continue
            
            # Check if clusters are still connected
            gi = self.get_cluster(ci)
            gj = self.get_cluster(cj)
            edges = self.connecting_edges(gi, gj)
            
            if not edges:
                continue
            
            # Perform the merge
            self.merge_clusters(ci, cj)
            
            # Update priority queue for the merged cluster
            self.update_priority_queue_after_merge(priority_queue, ci, cj)
            
            successful_merges += 1
            current_cluster_count = len(np.unique(self.df['cluster']))
            
            # Calculate and store NMI and Accuracy
            current_nmi = self.calculate_current_nmi()
            current_accuracy = self.calculate_current_accuracy()
            self.nmi_history.append((current_cluster_count, current_nmi))
            self.accuracy_history.append((current_cluster_count, current_accuracy))
            
            if verbose:
                print(f"Merged clusters {cj} -> {ci}, Clusters: {current_cluster_count}, NMI: {current_nmi:.4f}, Accuracy: {current_accuracy:.4f}")
        
        # Phase 2: Handle disconnected components
        if current_cluster_count > 1:
            if verbose:
                print(f"Handling {current_cluster_count} disconnected components...")
            
            components = self.find_disconnected_components()
            component_merges = self.merge_within_components(components, verbose)
            successful_merges += component_merges
            
            current_cluster_count = len(np.unique(self.df['cluster']))
        
        pbar.close()
        
        final_cluster_count = len(np.unique(self.df['cluster']))
        if verbose:
            print(f"Completed merging: {successful_merges} merges performed")
            print(f"Final cluster count: {final_cluster_count}")
        
        return successful_merges > 0

#==============================================================================
# Updated merge_clusters function to support complete merging
#==============================================================================

def merge_clusters(graph, df, alpha, beta, target_clusters, method='optimal', 
                  verbose=False, true_labels=None, complete_merge=False):
    """
    Merge clusters with option for complete merging to 1 cluster and NMI tracking.
    
    Parameters:
    - complete_merge: If True, merge all the way to 1 cluster regardless of target_clusters
    - true_labels: True labels for NMI calculation during merging
    """
    if method == 'optimal':
        # Use optimal priority queue merging
        clusterer = OptimizedGraphClusterer(graph, df, alpha, beta, true_labels)
        
        if verbose:
            current_clusters = len(np.unique(df["cluster"]))
            if complete_merge:
                print(f"Starting complete merging: {current_clusters} -> 1 cluster")
            else:
                print(f"Starting optimal merging: {current_clusters} -> {target_clusters} clusters")
        
        if complete_merge:
            clusterer.complete_merge_to_one(verbose=verbose)
        else:
            clusterer.optimized_merge_best(target_clusters, verbose=verbose)
        
        final_clusters = len(np.unique(df["cluster"]))
        
        if verbose:
            print(f"Merging completed: {final_clusters} final clusters")
        
        # Return results including NMI history
        max_nmi = 0.0
        max_nmi_clusters = final_clusters
        max_accuracy = 0.0
        max_accuracy_clusters = final_clusters
        
        if hasattr(clusterer, 'nmi_history') and clusterer.nmi_history:
            max_nmi_entry = max(clusterer.nmi_history, key=lambda x: x[1])
            max_nmi_clusters, max_nmi = max_nmi_entry
        
        if hasattr(clusterer, 'accuracy_history') and clusterer.accuracy_history:
            max_accuracy_entry = max(clusterer.accuracy_history, key=lambda x: x[1])
            max_accuracy_clusters, max_accuracy = max_accuracy_entry
        
        return final_clusters, max_nmi, max_nmi_clusters, max_accuracy, max_accuracy_clusters
    
    else:
        raise ValueError(f"Unknown merge method: {method}. Must be 'optimal'")

def normalize_cluster_labels(df):
    """Renumber cluster labels to be consecutive integers starting from 1"""
    result = df.copy()
    clusters = list(pd.DataFrame(df["cluster"].value_counts()).index)
    
    # Renumber clusters from 1 to n
    for i, cluster_id in enumerate(clusters, 1):
        result.loc[df["cluster"] == cluster_id, "cluster"] = i
    
    return result

#==============================================================================
# Main Function - Run_Ch2++
#==============================================================================

# For \t : UCI Datasets : explicitly tell #dimensions here in parameters
def read_fixed_columns_csv(filepath, expected_cols=8): 
    data = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            cleaned_line = re.sub(r'\t+', '\t', line.strip())
            parts = cleaned_line.split('\t')
            
            if len(parts) > expected_cols:
                parts = parts[:expected_cols]
            elif len(parts) < expected_cols:
                continue
            
            data.append(parts)
    
    return pd.DataFrame(data)

def Run_Ch2(path, log_type, r, alpha, beta,
            knn_type='symmetric', merge_method='optimal', logger=None):
    
    start_time = time.time()
    
    # Load & prepare Olivetti faces dataset
    # from sklearn.datasets import fetch_olivetti_faces
    # olivetti = fetch_olivetti_faces(shuffle=True, random_state=42)
    # X, y = olivetti.data, olivetti.target

    # # Create DataFrame
    # df = pd.DataFrame(X)
    # df['target'] = y
    # dataset = df.copy()

    # # Set up for Olivetti faces
    # feature_columns = dataset.columns[:-1].tolist()
    # num_classes = 40  # Olivetti has 40 classes
    # true_labels = dataset['target']

    # Other UCI Datasets
    #df = read_fixed_columns_csv(path)
    #df.columns = [0,1,2,3,4,5,6,7]      # seeds, wil
    
    # Load and prepare dataset - Generic
    # 7. UCI Real World
    # df = pd.read_csv(path)
    #df = pd.read_csv(path, header=None)
    
    #df.columns = df.iloc[0]
    #df = df[1:].reset_index(drop=True)
    #df.columns = df.columns.str.lower()
    
    # df = df.dropna()
    #df = df.apply(pd.to_numeric, errors='ignore')
    # df = df.apply(pd.to_numeric, errors='coerce') # for int stored as str in csv
    # dataset = df.copy()

    # Load and prepare dataset - Generic
    df = pd.read_csv(path, header=None)
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df.columns = df.columns.str.lower()
    df = df.apply(pd.to_numeric, errors='ignore')
    dataset = df.copy()

    # 1. Benchmark Datasets, 2. Pendigits, 3. Cytof.one
    feature_columns = dataset.columns[:-1].tolist()
    num_classes = dataset[dataset.columns[-1]].nunique()
    true_labels = dataset[dataset.columns[-1]]

    # 4. Cytof.h1, 5. Cytof.h2
    # feature_columns = ['cd45ra', 'cd133', 'cd19', 'cd22', 'cd11b', 'cd4', 'cd8', 'cd34', 'flt3', 'cd20','cxcr4', 'cd235ab', 'cd45', 'cd123', 'cd321', 'cd14', 'cd33', 'cd47', 'cd11c', 'cd7','cd15', 'cd16', 'cd44', 'cd38', 'cd13', 'cd3', 'cd61', 'cd117', 'cd49d', 'hla-dr','cd64', 'cd41']
    # num_classes = dataset[dataset.columns[-2]].nunique()
    # true_labels = dataset[dataset.columns[-2]]
    
    # 6. mnist
    #feature_columns = dataset.columns[1:].tolist()
    #num_classes = dataset[dataset.columns[0]].nunique()
    #true_labels = dataset[dataset.columns[0]] 

    n = len(dataset)
    df = dataset[feature_columns]

    # Initial partition count = sqrt(n/2)
    part_count = math.floor(math.sqrt(n / 2))
    
    # Dictionary to store experiment results
    result_dict = {
        'dataset': os.path.basename(path),
        '#classes': num_classes,
        'log_type': log_type,
        'r': r,
        'size (n)': n,
        'knn_type': knn_type,
        'merge_method': merge_method
    }
    
    # Calculate k for k-NN graph based on log_type and r
    if log_type == "ln": base = math.log(n)
    elif log_type == "log10": base = math.log10(n)
    elif log_type == "log2": base = math.log2(n)
    else: raise ValueError(f"Unknown log_type {log_type!r}")
    knn = r * int(base)
    
    # Update result dictionary
    result_dict['knn'] = knn
    result_dict['m (part)'] = part_count
    
    #==========================================================================
    # Step 1: Build k-NN graph
    #==========================================================================
    
    # Build the k-NN graph using unified function
    graph = build_knn_graph_annoy(df, knn, knn_type, n_trees=knn, verbose=True)
    
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    node_positions = nx.get_node_attributes(graph, "pos")
    
    # Update result dictionary
    result_dict['num_nodes'] = num_nodes
    result_dict['num_edges'] = num_edges
    
    #==========================================================================
    # Step 2: Initial partitioning and refinement
    #==========================================================================
    
    # Execute complete optimized partitioning
    graph, refined_partitions, partition_time, hmetis_partitions = \
        complete_optimized_partitioning(graph, part_count, df, verbose=True)
    final_part_count = len(refined_partitions)
    
    # Update result dictionary
    result_dict['Partitioned (sec)'] = f"{partition_time:.2f}"
    result_dict['Partitions after hMETIS'] = hmetis_partitions
    result_dict['Partitions after Flood-Fill'] = final_part_count
    
    #==========================================================================
    # Step 3: Complete merge to 1 cluster with NMI tracking
    #==========================================================================
    
    merge_start = time.time()
    
    # Use complete merging to 1 cluster
    final_clusters, max_nmi, max_nmi_clusters, max_accuracy, max_accuracy_clusters = merge_clusters(
        graph, df, alpha, beta, 1, 
        method=merge_method, verbose=True, 
        true_labels=true_labels, complete_merge=True
    )
    
    # Update result dictionary
    result_dict['alpha'] = alpha
    result_dict['beta'] = beta
    result_dict['Number of final clusters after merging'] = final_clusters

#==========================================================================
# Evaluate clustering results
#==========================================================================
    
    # Normalize cluster labels
    normalized_df = normalize_cluster_labels(df)
    predicted_labels = normalized_df["cluster"].to_numpy()
    
    # Calculate evaluation metrics
    final_nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    final_accuracy = hungarian_accuracy(true_labels, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    
    total_time = time.time() - start_time
    
    # Update result dictionary with NMI tracking results
    result_dict['NMI-Score'] = f"{final_nmi:.4f}"
    result_dict['Max NMI Score'] = f"{max_nmi:.4f}"
    result_dict['Max NMI at Clusters'] = max_nmi_clusters
    result_dict['Accuracy'] = f"{final_accuracy:.4f}"
    result_dict['Max Accuracy Score'] = f"{max_accuracy:.4f}"
    result_dict['Max Accuracy at Clusters'] = max_accuracy_clusters
    result_dict['Adjusted Rand Index'] = f"{ari:.4f}"
    result_dict['Total execution time (sec)'] = f"{total_time:.2f}"
    
    # Log results if logger is provided
    if logger is not None:
        logger.log_result(result_dict)
        logger.print_table(result_dict)
    
    return result_dict

# ==============================================================================
# Run experiments with both methods and k-NN types
# ==============================================================================

if __name__ == "__main__":
    # Set the path to your datasets
    dataset_dir = "Datasets/Ch2-Real-World"
    
    # Initialize logger
    logger = ResultLogger("Ch2++.csv")

    # Get list of dataset files (ignoring hidden files)
    dataset_files = [os.path.join(dataset_dir, fn) 
                   for fn in os.listdir(dataset_dir) 
                   if not fn.startswith(".")]

    # Define experiment parameters
    #log_types = ["ln", "log10", "log2"]
    log_types = ["log2"] 
    #r_values = [2**i for i in range(4)]  # [1, 2, 4, 8]
    r_values = [2]
    alpha_values = [2]  # Alpha values varies from 1 to 4
    beta_values = [1]   # Beta values varies from 1 to 4
    knn_types = ['symmetric']  # ['symmetric', 'asymmetric']
    merge_methods = ['optimal']  # ['naive', 'optimal']

    # Run Ch2++ on each dataset with different parameter combinations - Generic
    for dataset_path in dataset_files:
      for log_type in log_types:
          for r in r_values:
              for alpha in alpha_values:
                  for beta in beta_values:
                      for knn_type in knn_types:
                          for merge_method in merge_methods:
                              try:
                                  Run_Ch2(dataset_path, log_type, r, alpha, beta, 
                                         knn_type, merge_method, logger)
                              except Exception as e:
                                  print(f"Error processing {dataset_path} with {knn_type}-{merge_method}: {e}")
                                  continue
    
    # For Olivetti-Faces Run -
    # for knn_type in knn_types:
    #     for log_type in log_types:
    #         for r in r_values:
    #             skip_combination = False
    #             for alpha in alpha_values:
    #                 if skip_combination:
    #                     break
    #                 for beta in beta_values:
    #                     if skip_combination:
    #                         break
    #                     for merge_method in merge_methods:
    #                         try:
    #                             result = Run_Ch2('olivetti_faces', log_type, r, alpha, beta, 
    #                                             knn_type, merge_method, logger)
    #                             if result is None:
    #                                 print(f"Skipping remaining alpha-beta combinations for {knn_type}, {log_type}, r={r} due to disconnected graph")
    #                                 skip_combination = True
    #                                 break
    #                         except Exception as e:
    #                             print(f"Error processing {'olivetti_faces'} with {knn_type}-{merge_method}, {log_type}, r={r}: {e}")
    #                             print(f"Skipping remaining alpha-beta combinations for this configuration")
    #                             skip_combination = True
    #                             break
