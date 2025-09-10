# ==============================================================================
# Chameleon2 Equivalent: Ch2 Java Replica Variant
# ==============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
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
from annoy import AnnoyIndex

# ==============================================================================
# Logger class for capturing experiment results
# ==============================================================================
class ResultLogger:
    def __init__(self, output_path="Ch2Run.csv"):
        """
        Initialize result logger to store experiment metrics.
        
        Parameters:
        - output_path: Path to CSV file for saving results
        """
        self.output_path = output_path
        self.results = []
        
        # Create file with headers if it doesn't exist (modify the below as per diff det of experiments)
        if not os.path.exists(output_path):
            headers = [
                'dataset', '#classes', 'log_type', 'r', 'size (n)', 'knn', 'knn_type', 'merge_method',
                'p_max', 'num_nodes', 'num_edges', 'Partitioned (sec)', 'Partitions after FM', 
                'Partitions after Flood-Fill', 'alpha', 'beta', 'Number of final clusters after merging', 
                'Max-NMI-Score','Max NMI at Clusters', 'Total execution time (sec)'
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
# K-NN GRAPH - Brute Force
# ==============================================================================

def knn_graph_asymmetric_brute_force(df, knn, verbose=False):
    """Build asymmetric k-NN graph using efficient brute force"""
    points = np.array([p[1:] for p in df.itertuples()])
    
    if verbose:
        print(f"Building asymmetric k-NN (knn = {knn})...")
    
    g = nx.Graph()
    g.add_nodes_from(range(len(points)))
    
    iterator = tqdm(range(len(points)), total=len(points)) if verbose else range(len(points))
    
    for i in iterator:
        distances = np.linalg.norm(points - points[i], axis=1)
        distances[i] = np.inf  # Exclude self by setting to infinity
        
        # Get k smallest - guaranteed to exclude self
        closest_indices = np.argpartition(distances, knn)[:knn]
        
        for j in closest_indices:
            distance = distances[j]
            if distance > 0:
                weight = max(1, min(1000, int(1000.0 / (distance + 1e-10))))
                g.add_edge(i, j, weight=weight)
        
        g.nodes[i]['pos'] = points[i]
    
    g.graph['edge_weight_attr'] = 'weight'
    
    if verbose:
        print(f"Asymmetric k-NN graph created with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
        if g.number_of_edges() > 0:
            print(f"Weight range: {min(d['weight'] for u, v, d in g.edges(data=True))} to {max(d['weight'] for u, v, d in g.edges(data=True))}")
    
    return g


def knn_graph_symmetric_brute_force(df, knn, verbose=False):
    """Build symmetric k-NN graph using efficient brute force"""
    points = np.array([p[1:] for p in df.itertuples()])
    
    if verbose:
        print(f"Building symmetric k-NN (knn = {knn})...")
    
    # First pass: Build k-NN sets for each node for symmetric check
    knn_sets = {}
    if verbose:
        print("First pass: Building k-NN sets...")
    
    first_pass_iterator = tqdm(range(len(points)), total=len(points)) if verbose else range(len(points))
    for i in first_pass_iterator:
        distances = np.linalg.norm(points - points[i], axis=1)
        distances[i] = np.inf  # Exclude self
        
        closest_indices = np.argpartition(distances, knn)[:knn]
        knn_sets[i] = set(closest_indices)
    
    # Second pass: Build graph with symmetry constraint
    g = nx.Graph()
    g.add_nodes_from(range(len(points)))
    
    if verbose:
        print("Second pass: Building symmetric graph...")
    
    second_pass_iterator = tqdm(range(len(points)), total=len(points)) if verbose else range(len(points))
    for i in second_pass_iterator:
        distances = np.linalg.norm(points - points[i], axis=1)
        distances[i] = np.inf  # Exclude self
        
        closest_indices = np.argpartition(distances, knn)[:knn]
        
        for j in closest_indices:
            # Only add edge if mutual k-NN (symmetric condition)
            if i in knn_sets[j]:
                distance = distances[j]
                if distance > 0:
                    weight = max(1, min(1000, int(1000.0 / (distance + 1e-10))))
                    g.add_edge(i, j, weight=weight)
        
        g.nodes[i]['pos'] = points[i]
    
    g.graph['edge_weight_attr'] = 'weight'
    
    if verbose:
        print(f"Symmetric k-NN graph created with {g.number_of_nodes()} nodes and {g.number_of_edges()} edges")
        if g.number_of_edges() > 0:
            print(f"Weight range: {min(d['weight'] for u, v, d in g.edges(data=True))} to {max(d['weight'] for u, v, d in g.edges(data=True))}")
    
    return g

def build_knn_graph(df, knn, knn_type='symmetric', verbose=False):
    """
    Build k-NN graph with specified type.
    
    Parameters:
    - df: DataFrame with features
    - knn: Number of nearest neighbors
    - knn_type: 'symmetric' or 'asymmetric'
    - verbose: Whether to display progress
    
    Returns:
    - NetworkX graph
    """
    start_time = time.time()

    if knn_type == 'symmetric':
        graph = knn_graph_symmetric_brute_force(df, knn, verbose)
    elif knn_type == 'asymmetric':
        graph = knn_graph_asymmetric_brute_force(df, knn, verbose)
    else:
        raise ValueError(f"Unknown knn_type: {knn_type}. Must be 'symmetric' or 'asymmetric'")

    print(f"Exact k-NN graph construction time: {time.time() - start_time:.7f} seconds")
    return graph

# ==============================================================================
# PARTITIONING (FM) AND REFINEMENT FUNCTIONS (Flood-Fill)
# ==============================================================================

class Vertex:
    """Vertex class matching the Ch2 implementation"""
    def __init__(self, node_id):
        self.node_id = node_id
        self.cluster = 0
        self.used = False
        self.difference = 0
        self.previous = None
        self.next = None

class FiducciaMattheyses:
    """FM implementation exactly matching the Ch2 reference"""
    
    def __init__(self, iteration_limit=20):
        self.iteration_limit = iteration_limit
        self.vertices = {}
        self.graph = None
        self.node_count = 0
        self.max_degree = 0
        self.difference_buckets = None
        self.swap_history = []
        self.swap_history_cost = []
    
    def find_max_degree(self):
        """Find maximum node degree in the graph"""
        self.max_degree = 0
        for node in self.graph.nodes():
            degree = self.graph.degree(node)
            if degree > self.max_degree:
                self.max_degree = degree
    
    def create_vertices(self):
        """Create vertex objects for all nodes"""
        self.vertices = {}
        for i, node in enumerate(self.graph.nodes()):
            self.vertices[node] = Vertex(node)
    
    def create_initial_partition(self):
        """Randomly assigns nodes to clusters at the beginning"""
        nodes = list(self.graph.nodes())
        mid = self.node_count // 2
        
        # Assign first half to cluster 0, second half to cluster 1
        for i, node in enumerate(nodes):
            if i < mid:
                self.vertices[node].cluster = 0
            else:
                self.vertices[node].cluster = 1
    
    def prepare_difference_buckets(self):
        """Initialize difference buckets array"""
        # Buckets range from -max_degree to +max_degree
        # Index 0 represents difference -max_degree, index 2*max_degree represents +max_degree
        self.difference_buckets = [None] * (2 * self.max_degree + 1)
    
    def add_into_bucket(self, difference, vertex):
        """Add vertex into bucket with the given difference"""
        position = difference + self.max_degree
        if position < 0 or position >= len(self.difference_buckets):
            return  # Skip invalid positions
            
        if self.difference_buckets[position] is None:
            self.difference_buckets[position] = vertex
            vertex.next = None
        else:
            self.difference_buckets[position].previous = vertex
            vertex.next = self.difference_buckets[position]
            self.difference_buckets[position] = vertex
        vertex.previous = None
    
    def remove_from_bucket(self, vertex):
        """Remove given vertex from the difference bucket"""
        position = vertex.difference + self.max_degree
        if position < 0 or position >= len(self.difference_buckets):
            return
            
        if vertex.previous is not None:
            vertex.previous.next = vertex.next
        else:
            self.difference_buckets[position] = vertex.next
        
        if vertex.next is not None:
            vertex.next.previous = vertex.previous
    
    def compute_differences(self):
        """Compute differences of all nodes"""
        self.prepare_difference_buckets()
        
        for node in self.graph.nodes():
            vertex = self.vertices[node]
            difference = 0
            
            # Get all neighbors of this node
            for neighbor in self.graph.neighbors(node):
                neighbor_vertex = self.vertices[neighbor]
                if vertex.cluster != neighbor_vertex.cluster:
                    difference += 1  # External connection
                else:
                    difference -= 1  # Internal connection
            
            self.add_into_bucket(difference, vertex)
            vertex.difference = difference
    
    def find_best_vertex(self, cluster):
        """
        Find best vertex to swap from the given cluster.
        If no vertex from given cluster left, return best from other cluster.
        """
        # Start from highest difference (best gain)
        i = 2 * self.max_degree
        while i >= 0:
            if i < len(self.difference_buckets) and self.difference_buckets[i] is not None:
                item = self.difference_buckets[i]
                while item is not None:
                    if item.cluster == cluster and not item.used:
                        return item
                    item = item.next
            i -= 1
        
        # If no vertex found in the preferred cluster, try the other cluster
        other_cluster = 1 if cluster == 0 else 0
        return self.find_best_vertex(other_cluster) if cluster != other_cluster else None
    
    def add_into_history(self, i, vertex):
        """Add vertex and its cost to the swapping history"""
        if i >= len(self.swap_history):
            self.swap_history.extend([None] * (i - len(self.swap_history) + 1))
            self.swap_history_cost.extend([0] * (i - len(self.swap_history_cost) + 1))
        
        self.swap_history[i] = vertex
        self.swap_history_cost[i] = vertex.difference
        vertex.used = True
    
    def update_differences(self, vertex):
        """
        Change positions in difference_buckets of vertices neighboring the given vertex
        """
        for neighbor_node in self.graph.neighbors(vertex.node_id):
            neighbor = self.vertices[neighbor_node]
            
            if neighbor.used:
                continue
            
            # Remove from current bucket
            self.remove_from_bucket(neighbor)
            
            if neighbor.cluster == vertex.cluster:
                # Same cluster - decrease difference by 2
                neighbor.difference += 2
            else:
                # Different cluster - increase difference by 2  
                neighbor.difference -= 2
            
            # Add back to appropriate bucket
            self.add_into_bucket(neighbor.difference, neighbor)
    
    def find_best_swaps(self):
        """
        Find how many swaps should be done to achieve best difference sum.
        If two or more swap sequences have same gain, choose better balanced one.
        """
        max_difference = 0
        difference_sum = 0
        max_difference_index = -1
        
        for i in range(len(self.swap_history_cost)):
            difference_sum += self.swap_history_cost[i]
            # Keep the bisection balanced, only swap pairs (i % 2 == 1)
            if difference_sum > max_difference and i % 2 == 1:
                max_difference = difference_sum
                max_difference_index = i
        
        return max_difference_index
    
    def swap_up_to_best_index(self):
        """
        Make the best possible sequence of swaps and return index of last vertex swapped.
        If index is -1, no vertices were swapped.
        """
        index = self.find_best_swaps()
        
        if index >= 0:
            for i in range(index + 1):
                if i < len(self.swap_history) and self.swap_history[i] is not None:
                    vertex = self.swap_history[i]
                    vertex.cluster = 1 if vertex.cluster == 0 else 0
        
        return index
    
    def reset(self):
        """Prepare algorithm for the next iteration"""
        for vertex in self.vertices.values():
            vertex.used = False
        self.swap_history = []
        self.swap_history_cost = []
    
    def minimize_costs(self):
        self.compute_differences()
        
        for i in range(self.node_count):
            # Alternate between clusters to maintain balance
            cluster_preference = i % 2
            best_vertex = self.find_best_vertex(cluster_preference)
            
            if best_vertex is None:
                break
                
            self.add_into_history(i, best_vertex)
            self.update_differences(best_vertex)
            self.remove_from_bucket(best_vertex)
        
        return self.swap_up_to_best_index()
    
    def bisect(self, graph, max_partition_size=15):
        """
        Main bisection method,
        Returns list of two clusters (lists of nodes)
        """
        # Initialize
        self.graph = graph
        self.node_count = graph.number_of_nodes()
        
        if self.node_count == 0:
            return [[], []]
        
        self.find_max_degree()
        self.create_vertices()
        self.create_initial_partition()
        
        iteration_counter = 0
        
        # Repeat until no better swap can be done
        while iteration_counter < self.iteration_limit:
            iteration_counter += 1
            index = self.minimize_costs()
            self.reset()
            if index == -1:  # No improvement
                break
        
        # Create node clusters
        cluster_0 = []
        cluster_1 = []
        
        for node, vertex in self.vertices.items():
            if vertex.cluster == 0:
                cluster_0.append(node)
            else:
                cluster_1.append(node)
        
        return [cluster_0, cluster_1]

def recursive_bisection_aligned(graph, max_partition_size):
    """
    Recursive bisection implementation
    """
    if graph.number_of_nodes() <= max_partition_size:
        return [set(graph.nodes())]
    
    # Create FM instance
    fm = FiducciaMattheyses()
    
    # Bisect current graph
    clusters = fm.bisect(graph, max_partition_size)
    
    result = []
    
    # Process each partition recursively
    for cluster_nodes in clusters:
        if len(cluster_nodes) <= max_partition_size:
            result.append(set(cluster_nodes))
        else:
            # Create subgraph for this cluster
            subgraph = graph.subgraph(cluster_nodes).copy()
            # Recursively partition
            sub_partitions = recursive_bisection_aligned(subgraph, max_partition_size)
            result.extend(sub_partitions)
    
    return result

def refine_partitions(graph, partition_indices):
    """
    Refine partitions by finding connected components within each partition.
    """
    refined_partitions = []
    for cluster_nodes in partition_indices:
        if not cluster_nodes:  # Skip empty partitions
            continue
        # Extract subgraph of this partition
        subgraph = graph.subgraph(cluster_nodes)
        # Find connected components in this subgraph
        connected_components = list(nx.connected_components(subgraph))
        refined_partitions.extend(connected_components)
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

def complete_fm_partitioning(graph, max_partition_size, df=None, verbose=False):
    """
    Complete FM partitioning with flood fill refinement.
    
    Args:
        graph: NetworkX graph to partition
        max_partition_size: Maximum allowed partition size
        df: Optional dataframe to update with cluster assignments
        verbose: Whether to print progress information
    
    Returns:
        tuple: (graph, refined_partitions, fm_partition_time, initial_partition_count)
    """
    total_start = time.time()
    
    if verbose:
        print("Begin FM Partitioning...")
    
    if max_partition_size <= 0:
        raise ValueError("Max partition size must be positive")
    if max_partition_size >= graph.number_of_nodes():
        if verbose:
            print(f"Warning: max_partition_size={max_partition_size} >= number of nodes. Each node will be its own partition.")
    
    try:
        # Step 1: Initial FM partitioning
        fm_start = time.time()
        final_partitions = recursive_bisection_aligned(graph, max_partition_size)
        
        # Create initial partition assignments
        parts = [0] * graph.number_of_nodes()
        node_to_index = {node: i for i, node in enumerate(graph.nodes())}
        
        for partition_id, partition_nodes in enumerate(final_partitions):
            for node in partition_nodes:
                if node in node_to_index:
                    parts[node_to_index[node]] = partition_id
        
        fm_partition_time = time.time() - fm_start
        
        if verbose:
            print(f"FM partitioning time: {fm_partition_time:.7f} seconds")
            print(f"Initial partitions: {len(final_partitions)}")
        
        # Step 2: Flood fill refinement
        flood_fill_start = time.time()
        
        # Group nodes by partition
        partition_groups = defaultdict(list)
        for node_idx, partition in enumerate(parts):
            node = list(graph.nodes())[node_idx]
            partition_groups[partition].append(node)
        
        # Convert to list format for refine_partitions function
        partition_indices = [partition_groups[i] for i in sorted(partition_groups.keys())]
        
        # Refine partitions using flood fill
        refined_partitions = refine_partitions(graph, partition_indices)
        
        # Convert sets to lists if needed
        refined_partitions = [list(partition) if isinstance(partition, set) else partition 
                             for partition in refined_partitions]
        
        flood_fill_time = time.time() - flood_fill_start
        
        if verbose:
            print(f"Flood-Fill time: {flood_fill_time:.7f} seconds")
            print(f"Refined partitions: {len(refined_partitions)}")
        
        # Step 3: Update graph and dataframe
        final_partition_count = update_graph_and_dataframe(graph, df, refined_partitions, verbose=verbose)
        
        total_time = time.time() - total_start
        
        if verbose:
            print(f"Total FM partitioning time: {total_time:.7f} seconds")
        
    except Exception as e:
        if verbose:
            print(f"FM partitioning failed: {e}")
        # Fallback to simple alternating partition
        parts = [i % 2 for i in range(len(graph.nodes()))]
        refined_partitions = [[], []]
        for i, node in enumerate(graph.nodes()):
            refined_partitions[parts[i]].append(node)
        fm_partition_time = 0.0
        final_partition_count = update_graph_and_dataframe(graph, df, refined_partitions, verbose=verbose)
    
    return graph, refined_partitions, fm_partition_time, len(set(parts))

# ==============================================================================
# Ch2-COMPATIBLE MERGING CLASS - Matching Original Paper Implementation
# ==============================================================================

import heapq
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score

class OptimizedChameleonMerger:
    """
    Optimized Chameleon 2 clustering implementation with O(m² log m) complexity.
    Uses incremental updates for true linear update time per similarity recalculation.
    """
    def __init__(self, graph, df, alpha, beta, true_labels=None, mfact=1000):
        self.graph = graph
        self.df = df
        self.alpha = alpha  # Closeness priority (default: 2.0)
        self.beta = beta    # Inter-connectivity priority (default: 1.0)
        self.mfact = mfact  # Factor for small clusters (default: 10^3)
        self.true_labels = true_labels
        self.nmi_history = []
        
        # Initialize cluster mappings and precomputed metrics
        self.clusters = list(np.unique(self.df['cluster']))
        self.cluster_nodes = self._build_cluster_node_mapping()
        
        # Precompute and cache cluster metrics for incremental updates
        self.cluster_metrics = self._precompute_cluster_metrics()
        self.inter_cluster_metrics = self._precompute_inter_cluster_metrics()
        
        # Build similarity matrix using cached metrics
        self.similarity_matrix = self._build_similarity_matrix_optimized()
        
        # Priority queue for O(m² log m) merging
        self.merge_queue = []
        self._initialize_merge_queue()
    
    def _build_cluster_node_mapping(self):
        """Build mapping from cluster ID to list of nodes"""
        cluster_nodes = {}
        for cluster_id in self.clusters:
            cluster_nodes[cluster_id] = [
                node for node in self.graph.nodes() 
                if self.graph.nodes[node]['cluster'] == cluster_id
            ]
        return cluster_nodes
    
    def _precompute_cluster_metrics(self):
        """Precompute internal cluster metrics for incremental updates"""
        metrics = {}
        
        for cluster_id in self.clusters:
            nodes = self.cluster_nodes[cluster_id]
            internal_edges = self._get_internal_edges(nodes)
            
            # Cache computed values
            total_weight = sum(weight for _, _, weight in internal_edges)
            edge_count = len(internal_edges)
            
            metrics[cluster_id] = {
                'internal_edges': internal_edges,
                'total_weight': total_weight,  # s(Ci)
                'edge_count': edge_count,      # |ECi|
                'avg_weight': total_weight / edge_count if edge_count > 0 else 0.0,  # s̄(Ci)
                'nodes': nodes.copy()
            }
        
        return metrics
    
    def _precompute_inter_cluster_metrics(self):
        """Precompute inter-cluster connection metrics"""
        inter_metrics = {}
        
        for i, cluster_i in enumerate(self.clusters):
            for j, cluster_j in enumerate(self.clusters):
                if i < j:  # Only upper triangle
                    nodes_i = self.cluster_nodes[cluster_i]
                    nodes_j = self.cluster_nodes[cluster_j]
                    connecting_edges = self._get_connecting_edges(nodes_i, nodes_j)
                    
                    if connecting_edges:
                        total_weight = sum(weight for _, _, weight in connecting_edges)
                        edge_count = len(connecting_edges)
                        avg_weight = total_weight / edge_count
                        
                        inter_metrics[(cluster_i, cluster_j)] = {
                            'connecting_edges': connecting_edges,
                            'total_weight': total_weight,
                            'edge_count': edge_count,      # |ECi,j|
                            'avg_weight': avg_weight       # s̄(Ci,Cj)
                        }
        
        return inter_metrics
    
    def _get_connecting_edges(self, nodes_i, nodes_j):
        """Get edges connecting two clusters"""
        connecting_edges = []
        nodes_j_set = set(nodes_j)
        
        for node_i in nodes_i:
            if node_i in self.graph:
                for neighbor in self.graph[node_i]:
                    if neighbor in nodes_j_set:
                        weight = self.graph[node_i][neighbor].get('weight', 1)
                        connecting_edges.append((node_i, neighbor, weight))
        
        return connecting_edges
    
    def _get_internal_edges(self, nodes):
        """Get internal edges within a cluster"""
        internal_edges = []
        nodes_set = set(nodes)
        
        for node in nodes:
            if node in self.graph:
                for neighbor in self.graph[node]:
                    if neighbor in nodes_set and node < neighbor:  # Avoid duplicates
                        weight = self.graph[node][neighbor].get('weight', 1)
                        internal_edges.append((node, neighbor, weight))
        
        return internal_edges
    
    def _build_similarity_matrix_optimized(self):
        """Build similarity matrix using precomputed metrics"""
        matrix = {}
        
        for (cluster_i, cluster_j) in self.inter_cluster_metrics:
            similarity = self._calculate_ch2_similarity_cached(cluster_i, cluster_j)
            if similarity > 0:
                matrix[(cluster_i, cluster_j)] = similarity
        
        return matrix
    
    def _calculate_ch2_similarity_cached(self, cluster_i, cluster_j):
        """Calculate Ch2 similarity using cached metrics"""
        # Get cached metrics
        metrics_i = self.cluster_metrics[cluster_i]
        metrics_j = self.cluster_metrics[cluster_j]
        
        # Get inter-cluster metrics
        key = (min(cluster_i, cluster_j), max(cluster_i, cluster_j))
        if key not in self.inter_cluster_metrics:
            return 0.0  # No connection
        
        inter_metrics = self.inter_cluster_metrics[key]
        
        # Calculate RCL2 using cached values
        rcl2 = self._calculate_rcl2_cached(metrics_i, metrics_j, inter_metrics)
        
        # Calculate RIC2 using cached values
        ric2 = self._calculate_ric2_cached(metrics_i, metrics_j, inter_metrics)
        
        # Combined Ch2 similarity: RCL2^α * RIC2
        similarity = (rcl2 ** self.alpha) * ric2
        return similarity
    
    def _calculate_rcl2_cached(self, metrics_i, metrics_j, inter_metrics):
        """Calculate RCL2 using cached metrics"""
        ec_i = metrics_i['edge_count']
        ec_j = metrics_j['edge_count']
        
        # Handle special case: cluster with no internal edges
        if ec_i == 0 or ec_j == 0:
            s_between = inter_metrics['avg_weight']
            s_i = metrics_i['total_weight']
            s_j = metrics_j['total_weight']
            
            if (s_i + s_j) == 0:
                return self.mfact
            
            return self.mfact * s_between / (s_i + s_j)
        
        # Normal case: RCL2 = (|ECi| + |ECj|) * s̄(Ci,Cj) / (s(Ci) + s(Cj))
        s_between = inter_metrics['avg_weight']
        s_i = metrics_i['total_weight']
        s_j = metrics_j['total_weight']
        
        if (s_i + s_j) == 0:
            return 0.0
        
        rcl2 = (ec_i + ec_j) * s_between / (s_i + s_j)
        return rcl2
    
    def _calculate_ric2_cached(self, metrics_i, metrics_j, inter_metrics):
        """Calculate RIC2 using cached metrics"""
        ec_i = metrics_i['edge_count']
        ec_j = metrics_j['edge_count']
        ec_ij = inter_metrics['edge_count']
        
        # Handle special case: cluster with no internal edges
        if ec_i == 0 or ec_j == 0:
            return 1.0
        
        # Normal case: RIC2 = |ECi,j| / min{|ECi|, |ECj|} * ρ(Ci,Cj)^β
        min_edges = min(ec_i, ec_j)
        if min_edges == 0:
            return 1.0
        
        # Calculate ρ factor using cached average weights
        s_bar_i = metrics_i['avg_weight']
        s_bar_j = metrics_j['avg_weight']
        
        if s_bar_i == 0 and s_bar_j == 0:
            rho = 1.0
        elif s_bar_i == 0 or s_bar_j == 0:
            rho = 0.0
        else:
            rho = min(s_bar_i, s_bar_j) / max(s_bar_i, s_bar_j)
        
        # RIC2 = (|ECi,j| / min{|ECi|, |ECj|}) * ρ^β
        ric2 = (ec_ij / min_edges) * (rho ** self.beta)
        return ric2
    
    def _initialize_merge_queue(self):
        """Initialize priority queue with all valid cluster pairs"""
        self.merge_queue = []
        
        for (cluster_i, cluster_j), similarity in self.similarity_matrix.items():
            if similarity > 0:
                heapq.heappush(self.merge_queue, (-similarity, cluster_i, cluster_j))
    
    def _update_external_properties_incremental(self, merged_cluster, removed_cluster):
        """
        Incrementally update similarity matrix - TRUE LINEAR UPDATE TIME
        """
        # Remove old similarities involving the removed cluster
        keys_to_remove = []
        for key in self.similarity_matrix:
            if removed_cluster in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.similarity_matrix[key]
        
        # Remove inter-cluster metrics involving removed cluster
        inter_keys_to_remove = []
        for key in self.inter_cluster_metrics:
            if removed_cluster in key:
                inter_keys_to_remove.append(key)
        
        for key in inter_keys_to_remove:
            del self.inter_cluster_metrics[key]
        
        # Update similarities incrementally - O(m) operations
        for other_cluster in self.clusters:
            if other_cluster != merged_cluster and other_cluster != removed_cluster:
                # Incrementally compute new inter-cluster metrics
                self._update_inter_cluster_metrics_incremental(merged_cluster, other_cluster)
                
                # Calculate similarity using updated cached metrics
                similarity = self._calculate_ch2_similarity_cached(merged_cluster, other_cluster)
                
                if similarity > 0:
                    key = (min(merged_cluster, other_cluster), max(merged_cluster, other_cluster))
                    self.similarity_matrix[key] = similarity
                    heapq.heappush(self.merge_queue, (-similarity, merged_cluster, other_cluster))
    
    def _update_inter_cluster_metrics_incremental(self, cluster_merged, other_cluster):
        """
        Incrementally update inter-cluster metrics - CORE OPTIMIZATION
        """
        # Get nodes for both clusters
        nodes_merged = self.cluster_metrics[cluster_merged]['nodes']
        nodes_other = self.cluster_metrics[other_cluster]['nodes']
        
        # Calculate connecting edges (this is the only O(edges) operation needed)
        connecting_edges = self._get_connecting_edges(nodes_merged, nodes_other)
        
        if connecting_edges:
            total_weight = sum(weight for _, _, weight in connecting_edges)
            edge_count = len(connecting_edges)
            avg_weight = total_weight / edge_count
            
            key = (min(cluster_merged, other_cluster), max(cluster_merged, other_cluster))
            self.inter_cluster_metrics[key] = {
                'connecting_edges': connecting_edges,
                'total_weight': total_weight,
                'edge_count': edge_count,
                'avg_weight': avg_weight
            }
    
    def _merge_clusters_incremental(self, cluster_i, cluster_j):
        """
        Merge clusters with incremental metric updates
        """
        # Update DataFrame
        self.df.loc[self.df['cluster'] == cluster_j, 'cluster'] = cluster_i
        
        # Update graph nodes
        for node in self.graph.nodes():
            if self.graph.nodes[node]['cluster'] == cluster_j:
                self.graph.nodes[node]['cluster'] = cluster_i
        
        # Incrementally merge cluster metrics - CORE OPTIMIZATION
        metrics_i = self.cluster_metrics[cluster_i]
        metrics_j = self.cluster_metrics[cluster_j]
        
        # Merge internal edges and metrics
        merged_internal_edges = metrics_i['internal_edges'] + metrics_j['internal_edges']
        
        # Add edges that were connecting i and j (they become internal)
        key = (min(cluster_i, cluster_j), max(cluster_i, cluster_j))
        if key in self.inter_cluster_metrics:
            connecting_edges = self.inter_cluster_metrics[key]['connecting_edges']
            # Convert connecting edges to internal edge format
            for edge in connecting_edges:
                node1, node2, weight = edge
                if node1 < node2:
                    merged_internal_edges.append((node1, node2, weight))
                else:
                    merged_internal_edges.append((node2, node1, weight))
        
        # Update merged cluster metrics
        merged_total_weight = sum(weight for _, _, weight in merged_internal_edges)
        merged_edge_count = len(merged_internal_edges)
        merged_avg_weight = merged_total_weight / merged_edge_count if merged_edge_count > 0 else 0.0
        merged_nodes = metrics_i['nodes'] + metrics_j['nodes']
        
        self.cluster_metrics[cluster_i] = {
            'internal_edges': merged_internal_edges,
            'total_weight': merged_total_weight,
            'edge_count': merged_edge_count,
            'avg_weight': merged_avg_weight,
            'nodes': merged_nodes
        }
        
        # Remove metrics for removed cluster
        del self.cluster_metrics[cluster_j]
        
        # Update cluster node mapping
        self.cluster_nodes[cluster_i] = merged_nodes
        del self.cluster_nodes[cluster_j]
        
        # Remove merged cluster from cluster list
        self.clusters.remove(cluster_j)
        
        # Update external properties incrementally
        self._update_external_properties_incremental(cluster_i, cluster_j)
    
    def _single_merge(self):
        """Perform a single merge operation"""
        while self.merge_queue:
            neg_similarity, cluster_i, cluster_j = heapq.heappop(self.merge_queue)
            
            if cluster_i in self.clusters and cluster_j in self.clusters:
                similarity = -neg_similarity
                key = (min(cluster_i, cluster_j), max(cluster_i, cluster_j))
                
                if key in self.similarity_matrix and abs(self.similarity_matrix[key] - similarity) < 1e-10:
                    self._merge_clusters_incremental(cluster_i, cluster_j)
                    return True
        
        return False
    
    def _calculate_current_nmi(self):
        """Calculate NMI for current clustering state"""
        if self.true_labels is None:
            return 0.0
        
        normalized_df = normalize_cluster_labels(self.df)
        predicted_labels = normalized_df["cluster"].to_numpy()
        return normalized_mutual_info_score(self.true_labels, predicted_labels)
    
    def merge_to_target_clusters(self, target_clusters, verbose=False):
        """Merge clusters down to target number"""
        current_clusters = len(self.clusters)
        
        if verbose:
            print(f"Starting optimized merge from {current_clusters} to {target_clusters} clusters")
        
        initial_nmi = self._calculate_current_nmi()
        self.nmi_history = [(current_clusters, initial_nmi)]
        
        merge_count = 0
        max_merges = current_clusters - target_clusters
        
        with tqdm(total=max_merges, desc=f"Merging to {target_clusters} clusters") as pbar:
            while len(self.clusters) > target_clusters:
                if self._single_merge():
                    merge_count += 1
                    current_count = len(self.clusters)
                    
                    current_nmi = self._calculate_current_nmi()
                    self.nmi_history.append((current_count, current_nmi))
                    
                    pbar.update(1)
                    
                    if verbose:
                        print(f"Merge {merge_count}: {current_count} clusters, NMI: {current_nmi:.4f}")
                else:
                    if verbose:
                        print("No more valid merges possible")
                    break
        
        final_clusters = len(self.clusters)
        #if verbose:
        #    print(f"Optimized merging completed: {merge_count} merges, {final_clusters} final clusters")
        
        return merge_count > 0
    
    def complete_merge_to_one(self, verbose=False):
        """Complete merge to 1 cluster with comprehensive NMI tracking"""
        return self.merge_to_target_clusters(1, verbose)


def normalize_cluster_labels(df):
    """Renumber cluster labels to be consecutive integers starting from 1"""
    result = df.copy()
    clusters = list(pd.DataFrame(df["cluster"].value_counts()).index)
    
    for i, cluster_id in enumerate(clusters, 1):
        result.loc[df["cluster"] == cluster_id, "cluster"] = i
    
    return result


def merge_clusters(graph, df, alpha, beta, target_clusters, method='optimal', 
                  verbose=False, true_labels=None, complete_merge=False, mfact=1000):
    """
    Merge clusters using optimized Chameleon 2 implementation with incremental updates.
    """
    if method == 'optimal':
        # Use optimized merger with incremental updates
        merger = OptimizedChameleonMerger(graph, df, alpha, beta, true_labels, mfact)
        
        if verbose:
            current_clusters = len(np.unique(df["cluster"]))
            if complete_merge:
                print(f"Starting optimized Chameleon 2 complete merging: {current_clusters} -> 1 cluster")
            else:
                print(f"Starting optimized Chameleon 2 merging: {current_clusters} -> {target_clusters} clusters")
        
        if complete_merge:
            merger.complete_merge_to_one(verbose=verbose)
        else:
            merger.merge_to_target_clusters(target_clusters, verbose=verbose)
        
        final_clusters = len(np.unique(df["cluster"]))
        
        #if verbose:
        #    print(f"Optimized Chameleon 2 merging completed: {final_clusters} final clusters")
        
        # Return results including NMI history
        max_nmi = 0.0
        max_nmi_clusters = final_clusters
        
        if hasattr(merger, 'nmi_history') and merger.nmi_history:
            max_nmi_entry = max(merger.nmi_history, key=lambda x: x[1])
            max_nmi_clusters, max_nmi = max_nmi_entry
        
        return final_clusters, max_nmi, max_nmi_clusters
    
    else:
        raise ValueError(f"Unknown merge method: {method}. Must be 'optimal'")

# ==============================================================================
# Updated Main Function: Run_Ch2 
# ==============================================================================
def Run_Ch2(path, log_type, r, alpha, beta, knn_type='asymmetric', merge_method='optimal', logger=None):
    """
    Execute the Chameleon2 clustering algorithm on the given dataset.
    """
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
    # feature_columns = dataset.columns[1:].tolist()
    # num_classes = dataset[dataset.columns[0]].nunique()
    # true_labels = dataset[dataset.columns[0]] 

    n = len(dataset)
    df = dataset[feature_columns]

    # Initial partition count = sqrt(n/2)
    # part_count = math.floor(math.sqrt(n / 2))
    max_partition_size = max(5, n // 100)
    
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
    if   log_type == "ln":    base = math.log(n)
    elif log_type == "log10": base = math.log10(n)
    elif log_type == "log2":  base = math.log2(n)
    else: raise ValueError(f"Unknown log_type {log_type!r}")
    knn = r * int(base)
    # n_trees = knn
    
    # Update result dictionary
    result_dict['knn'] = knn
    result_dict['p_max'] = max_partition_size

    # ==============================================================================
    # Step 1: Build k-NN graph (UNIFIED)
    # ==============================================================================
    
    # Build the k-NN graph using unified function
    graph = build_knn_graph(df, knn, knn_type, verbose=True)
    
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    node_positions = nx.get_node_attributes(graph, "pos")
    
    # Update result dictionary
    result_dict['num_nodes'] = num_nodes
    result_dict['num_edges'] = num_edges

    # ==============================================================================
    # Step 2: Initial partitioning and refinement 
    # ==============================================================================
    
    # Execute complete optimized partitioning
    graph, refined_partitions, partition_time, fm_partitions = complete_fm_partitioning(graph, max_partition_size, df, verbose=True)
    final_part_count = len(refined_partitions)
    
    # Update result dictionary
    result_dict['Partitioned (sec)'] = f"{partition_time:.7f}"
    result_dict['Partitions after FM'] = fm_partitions
    result_dict['Partitions after Flood-Fill'] = final_part_count

    #==========================================================================
    # Step 3: Complete merge to 1 cluster with NMI tracking (UPDATED)
    #==========================================================================
    
    merge_start = time.time()
    
    # Use Ch2-compatible complete merging to 1 cluster
    final_clusters, max_nmi, max_nmi_clusters = merge_clusters(
        graph, df, alpha, beta, 1, 
        method=merge_method, verbose=True, 
        true_labels=true_labels, complete_merge=True
    )

    print(f"Ch2-compatible merging completed in {time.time() - merge_start:.7f} seconds")
    
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
    ari = adjusted_rand_score(true_labels, predicted_labels)
    
    total_time = time.time() - start_time
    
    # Update result dictionary with NMI tracking results
    result_dict['Max-NMI-Score'] = f"{max_nmi:.4f}"
    #result_dict['NMI Score'] = f"{final_nmi:.4f}"
    result_dict['Max NMI at Clusters'] = max_nmi_clusters
    #result_dict['Adjusted Rand Index'] = f"{ari:.4f}"
    result_dict['Total execution time (sec)'] = f"{total_time:.7f}"
    
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
    logger = ResultLogger("Ch2RunTempCh2-Real-World.csv")

    # Get list of dataset files (ignoring hidden files)
    dataset_files = [os.path.join(dataset_dir, fn) 
                  for fn in os.listdir(dataset_dir) 
                  if not fn.startswith(".")]

    # Define experiment parameters
    log_types = ["log2"] # ["ln", "log10", "log2"]
    r_values = [2]  # [1, 2, 4, 8]
    alpha_values = [2]  # [1, 2, 3, 4] Alpha values b/w 1 to 4
    beta_values = [1]   # [1, 2, 3, 4] Beta values b/w 1 to 4
    #knn_types = ['asymmetric']
    knn_types = ['symmetric']
    merge_methods = ['optimal']

    # Run Chameleon2 on each dataset with different parameter combinations - Generic
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
