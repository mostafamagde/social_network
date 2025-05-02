import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from community import best_partition
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import numpy as np
from collections import defaultdict
import math
import traceback
import sys
import random

class SocialNetworkAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Social Network Analysis Tool")
        self.root.geometry("1400x900")
        
        # Data storage
        self.G = None
        self.original_graph = None
        self.node_attributes = {}
        self.edge_attributes = pd.DataFrame()
        self.communities = {}
        
        # Visualization settings
        self.layout_pos = None
        self.current_layout = "spring"
        self.layout_seed = 42
        self.node_size = 300
        self.custom_color = "#1f78b4"
        self.node_shape = "o"
        self.edge_width = 1.0
        self.edge_style = "solid"
        self.arrow_size = 10
        self.node_label_visible = True
        self.node_label_size = 8
        
        # Initialize UI
        self.setup_ui()
        
    def setup_ui(self):
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', font=('Arial', 10), padding=5)
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        
        # Main frames
        self.control_frame_container = ttk.Frame(self.root)
        self.control_frame_container.pack(side=tk.LEFT, fill=tk.Y)
        
        # Create canvas and scrollbar for control frame
        self.control_canvas = tk.Canvas(self.control_frame_container)
        self.control_scrollbar = ttk.Scrollbar(self.control_frame_container, 
                                              orient="vertical", 
                                              command=self.control_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.control_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.control_canvas.configure(
                scrollregion=self.control_canvas.bbox("all")
            )
        )
        
        self.control_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)
        
        self.control_canvas.pack(side="left", fill="both", expand=True)
        self.control_scrollbar.pack(side="right", fill="y")
        
        # Control frame inside scrollable canvas
        self.control_frame = ttk.Frame(self.scrollable_frame, width=300, padding=10)
        self.control_frame.pack(fill=tk.BOTH, expand=True)
        
        # Visualization canvas
        self.figure = plt.figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Output frame
        self.output_frame = ttk.Frame(self.root)
        self.output_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        self.output_text = tk.Text(self.output_frame, wrap=tk.WORD, font=('Consolas', 10))
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(self.output_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.output_text.yview)
        
        # Build UI sections
        self.setup_data_loading()
        self.setup_visualization_controls()
        self.setup_analysis_controls()
        self.setup_filter_controls()
        self.setup_metrics_controls()
        
    def setup_data_loading(self):
        frame = ttk.LabelFrame(self.control_frame, text="Data Loading", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(frame, text="Load Nodes CSV", command=self.load_nodes).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Load Edges CSV", command=self.load_edges).pack(fill=tk.X, pady=2)
        
        self.directed_var = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Directed Graph", variable=self.directed_var).pack(anchor=tk.W)
        
    def setup_visualization_controls(self):
        frame = ttk.LabelFrame(self.control_frame, text="Visualization Settings", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        # Layout selection
        ttk.Label(frame, text="Layout:").pack(anchor=tk.W)
        self.layout_var = tk.StringVar(value="spring")
        ttk.Combobox(frame, textvariable=self.layout_var, 
                     values=["spring", "circular", "kamada-kawai", "spectral", "shell", "tree", "radial"]).pack(fill=tk.X)
        
        # Node size control
        ttk.Label(frame, text="Node Size:").pack(anchor=tk.W)
        self.size_scale = ttk.Scale(frame, from_=10, to=1000, value=300)
        self.size_scale.pack(fill=tk.X)
        
        # Color controls
        ttk.Label(frame, text="Node Color:").pack(anchor=tk.W)
        color_frame = ttk.Frame(frame)
        color_frame.pack(fill=tk.X)
        self.color_button = ttk.Button(color_frame, text="Choose Color", command=self.choose_node_color)
        self.color_button.pack(side=tk.LEFT)
        self.color_preview = tk.Canvas(color_frame, width=50, height=20, bg=self.custom_color)
        self.color_preview.pack(side=tk.LEFT, padx=5)
        
        # Color mapping selection
        self.color_var = tk.StringVar(value="manual")
        ttk.Combobox(frame, textvariable=self.color_var, 
                     values=["manual", "community", "degree", "betweenness", "pagerank"]).pack(fill=tk.X)
        
        # Shape controls
        ttk.Label(frame, text="Node Shape:").pack(anchor=tk.W)
        shape_frame = ttk.Frame(frame)
        shape_frame.pack(fill=tk.X)
        self.shape_var = tk.StringVar(value="o")
        shapes = [("Circle", "o"), ("Square", "s"), ("Triangle", "^"), ("Diamond", "D")]
        for text, shape in shapes:
            ttk.Radiobutton(shape_frame, text=text, variable=self.shape_var, 
                           value=shape).pack(side=tk.LEFT)
        
        # Edge styling
        ttk.Label(frame, text="Edge Style:").pack(anchor=tk.W)
        self.edge_style_var = tk.StringVar(value="solid")
        ttk.Combobox(frame, textvariable=self.edge_style_var,
                    values=["solid", "dashed", "dotted", "dashdot"]).pack(fill=tk.X)
        
        ttk.Label(frame, text="Edge Width:").pack(anchor=tk.W)
        self.edge_width_scale = ttk.Scale(frame, from_=1, to=10, value=1)
        self.edge_width_scale.pack(fill=tk.X)
        
        ttk.Button(frame, text="Update Visualization", command=self.update_visualization).pack(fill=tk.X, pady=5)
        ttk.Button(frame, text="Reapply Layout", command=self.force_new_layout).pack(fill=tk.X)
        
    def setup_analysis_controls(self):
        frame = ttk.LabelFrame(self.control_frame, text="Analysis Tools", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(frame, text="Run Louvain Community Detection", 
                  command=lambda: self.run_community_detection("louvain")).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Run Girvan-Newman Community Detection", 
                  command=lambda: self.run_community_detection("girvan")).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Compare Community Algorithms", 
                  command=self.compare_community_algorithms).pack(fill=tk.X, pady=2)
        
        ttk.Button(frame, text="Calculate PageRank", command=self.calculate_pagerank).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Calculate Betweenness Centrality", command=self.calculate_betweenness).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Calculate Clustering Coefficients", command=self.calculate_clustering).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Show Degree Distribution", command=self.plot_degree_distribution).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Calculate average Path Length", command=self.calculate_path_length).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Show Top 10 Degrees", command=self.show_top_degrees).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Show Top 10 Closeness", command=self.show_top_closeness).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Visualize Betweenness Distribution", command=lambda: self.visualize_metric_distribution("betweenness")).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Visualize Closeness Distribution", command=lambda: self.visualize_metric_distribution("closeness")).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Visualize Clustering Distribution", command=lambda: self.visualize_metric_distribution("clustering")).pack(fill=tk.X, pady=2)
        
    def setup_filter_controls(self):
        frame = ttk.LabelFrame(self.control_frame, text="Centrality Filters", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        self.filter_threshold = tk.DoubleVar(value=0.0)
        ttk.Entry(frame, textvariable=self.filter_threshold).pack(fill=tk.X)
        
        ttk.Button(frame, text="Filter by Degree", 
                  command=lambda: self.filter_centrality('degree')).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Filter by Betweenness", 
                  command=lambda: self.filter_centrality('betweenness')).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Filter by Closeness", 
                  command=lambda: self.filter_centrality('closeness')).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Filter by Harmonic", 
                  command=lambda: self.filter_centrality('harmonic')).pack(fill=tk.X, pady=2)
        
    def setup_metrics_controls(self):
        frame = ttk.LabelFrame(self.control_frame, text="Metrics", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(frame, text="Show Conductance", command=self.show_conductance).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Show Modularity", command=self.show_modularity).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Show NMI", command=self.show_nmi).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Show ARI", command=self.show_ari).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Show Harmonic Centrality", command=self.show_harmonic_centrality).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Clear All", command=self.clear_all).pack(fill=tk.X, pady=5)
        
    def log_message(self, message):
        """Add a message to the output log"""
        self.output_text.insert(tk.END, f"[LOG] {message}\n")
        self.output_text.see(tk.END)

    def show_error(self, title, message):
        """Show an error message"""
        error_msg = f"{title}:\n{message}"
        self.log_message(error_msg)
        messagebox.showerror(title, message)
        
    def choose_node_color(self):
        color = colorchooser.askcolor(title="Choose Node Color", initialcolor=self.custom_color)
        if color[1]:
            self.custom_color = color[1]
            self.color_preview.config(bg=self.custom_color)
            self.update_visualization()
        
    def load_nodes(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
            
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()
            
            if 'id' not in df.columns:
                messagebox.showerror("Error", "Node CSV must contain an 'id' column")
                return
                
            self.node_attributes = df.set_index('id').to_dict('index')
            messagebox.showinfo("Success", f"Loaded {len(self.node_attributes)} node attributes")
            self.layout_pos = None
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load nodes: {str(e)}")
    
    def load_edges(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return
            
        try:
            self.edge_attributes = pd.read_csv(file_path)
            self.edge_attributes.columns = self.edge_attributes.columns.str.lower()
            
            if not {'source', 'target'}.issubset(self.edge_attributes.columns):
                messagebox.showerror("Error", "Edge CSV must contain 'source' and 'target' columns")
                return
                
            self.create_graph()
            self.update_visualization()
            messagebox.showinfo("Success", f"Loaded {len(self.edge_attributes)} edges")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load edges: {str(e)}")
    
    def create_graph(self):
        try:
            self.G = nx.DiGraph() if self.directed_var.get() else nx.Graph()
            
            for node, attrs in self.node_attributes.items():
                self.G.add_node(node, **attrs)
                
            for _, row in self.edge_attributes.iterrows():
                edge_attrs = row.drop(['source', 'target']).to_dict()
                self.G.add_edge(row['source'], row['target'], **edge_attrs)
                
            self.original_graph = self.G.copy()
            self.layout_pos = None
            
        except Exception as e:
            messagebox.showerror("Error", f"Graph creation failed: {str(e)}")
            self.G = None
    
    def update_visualization(self):
        if self.G is None or len(self.G.nodes()) == 0:
            return
            
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        
        if self.layout_pos is None or self.current_layout != self.layout_var.get():
            self.current_layout = self.layout_var.get()
            self.layout_pos = self.calculate_layout()
            
        node_colors = self.get_node_colors()
        
        nx.draw_networkx_nodes(
            self.G, self.layout_pos, ax=ax,
            node_size=float(self.size_scale.get()),
            node_color=node_colors,
            cmap=plt.cm.tab20 if self.color_var.get() != "manual" else None,
            node_shape=self.shape_var.get()
        )
        
        edge_style = self.edge_style_var.get()
        edge_width = float(self.edge_width_scale.get())
        
        if self.directed_var.get():
            nx.draw_networkx_edges(
                self.G, self.layout_pos, ax=ax, 
                width=edge_width,
                style=edge_style,
                arrows=True,
                arrowsize=10
            )
        else:
            nx.draw_networkx_edges(
                self.G, self.layout_pos, ax=ax, 
                width=edge_width,
                style=edge_style
            )
        
        if len(self.G.nodes()) <= 50:
            nx.draw_networkx_labels(self.G, self.layout_pos, ax=ax, font_size=8)
        
        self.canvas.draw()
    
    def get_node_colors(self):
        color_by = self.color_var.get()
        
        if color_by == "manual":
            return self.custom_color
        elif color_by == "community":
            if not self.communities:
                return self.custom_color
            return [self.communities.get(node, 0) for node in self.G.nodes()]
        elif color_by == "degree":
            degrees = dict(self.G.degree())
            return [degrees[node] for node in self.G.nodes()]
        elif color_by == "betweenness":
            betweenness = nx.betweenness_centrality(self.G)
            return [betweenness[node] for node in self.G.nodes()]
        elif color_by == "pagerank":
            pagerank = nx.pagerank(self.G)
            return [pagerank[node] for node in self.G.nodes()]
        else:
            return self.custom_color
    
    def calculate_layout(self):
        layout = self.layout_var.get()
        
        try:
            if layout == "spring":
                return nx.spring_layout(self.G, seed=self.layout_seed)
            elif layout == "circular":
                return nx.circular_layout(self.G)
            elif layout == "kamada-kawai":
                return nx.kamada_kawai_layout(self.G)
            elif layout == "spectral":
                return nx.spectral_layout(self.G)
            elif layout == "shell":
                return nx.shell_layout(self.G)
            elif layout == "tree":
                return self.tree_layout()
            elif layout == "radial":
                return self.radial_layout()
            else:
                return nx.spring_layout(self.G, seed=self.layout_seed)
        except Exception as e:
            messagebox.showerror("Layout Error", str(e))
            return nx.spring_layout(self.G, seed=self.layout_seed)
    
    def tree_layout(self):
        """Hierarchical tree layout using BFS levels"""
        if len(self.G.nodes()) == 0:
            return {}
            
        root = max(self.G.degree(), key=lambda x: x[1])[0]
        levels = {root: 0}
        queue = [root]
        
        while queue:
            current = queue.pop(0)
            for neighbor in self.G.neighbors(current):
                if neighbor not in levels:
                    levels[neighbor] = levels[current] + 1
                    queue.append(neighbor)
        
        max_level = max(levels.values())
        pos = {}
        for node, level in levels.items():
            y = max_level - level
            x = len([n for n in levels if levels[n] == level and n <= node])
            pos[node] = (x - 0.5 * (len(levels) ** 0.5), y)
        
        return pos
    
    def radial_layout(self):
        """Radial layout with root at center"""
        if len(self.G.nodes()) == 0:
            return {}
            
        root = max(self.G.degree(), key=lambda x: x[1])[0]
        levels = nx.single_source_shortest_path_length(self.G, root)
        max_level = max(levels.values()) if levels else 1
        
        pos = {}
        for node, level in levels.items():
            angle = (node.__hash__() % 628) / 100  # Random angle between 0-6.28
            radius = level / max_level
            pos[node] = (radius * math.cos(angle), radius * math.sin(angle))
        
        return pos
    
    def force_new_layout(self):
        self.layout_pos = None
        self.update_visualization()
    
    def run_community_detection(self, method):
        if self.G is None:
            messagebox.showerror("Error", "No graph loaded")
            return
            
        try:
            if method == "louvain":
                partition = best_partition(self.G.to_undirected())
                self.communities = partition
                message = "Louvain communities detected"
            elif method == "girvan":
                communities = nx.community.girvan_newman(self.G.to_undirected())
                self.communities = {node: i for i, comm in enumerate(next(communities)) for node in comm}
                message = "Girvan-Newman communities detected"
            
            self.color_var.set("community")
            self.show_community_metrics()
            self.update_visualization()
            messagebox.showinfo("Success", message)
            
        except Exception as e:
            messagebox.showerror("Error", f"Community detection failed: {str(e)}")
    
    def show_community_metrics(self):
        if not self.communities:
            messagebox.showerror("Error", "Run community detection first")
            return
            
        metrics = []
        communities = defaultdict(list)
        for node, comm_id in self.communities.items():
            communities[comm_id].append(node)
        
        # Convert to list of sets for modularity calculation
        community_sets = [set(nodes) for nodes in communities.values()]
        
        metrics.append(f"Number of communities: {len(communities)}")
        try:
            mod = nx.community.modularity(self.G, community_sets)
            metrics.append(f"Modularity: {mod:.3f}")
        except Exception as e:
            metrics.append(f"Modularity calculation failed: {str(e)}")
        
        if hasattr(self, 'node_attributes') and 'class' in next(iter(self.node_attributes.values()), {}):
            try:
                ground_truth = {node: data.get('class', 0) for node, data in self.G.nodes(data=True)}
                detected = [self.communities[node] for node in self.G.nodes()]
                truth = [ground_truth[node] for node in self.G.nodes()]
                nmi = normalized_mutual_info_score(truth, detected)
                metrics.append(f"Normalized Mutual Information: {nmi:.3f}")
            except Exception as e:
                metrics.append(f"NMI calculation failed: {str(e)}")
        
        try:
            conductances = []
            for comm in community_sets:
                if len(comm) == 0:
                    continue
                cut = nx.cut_size(self.G, comm)
                volume = nx.volume(self.G, comm)
                total = self.G.number_of_edges() * 2  # Since each edge is counted twice
                conductance = cut / min(volume, total - volume) if min(volume, total - volume) != 0 else 0
                conductances.append(conductance)
            metrics.append(f"Average Conductance: {np.mean(conductances):.3f}")
        except Exception as e:
            metrics.append(f"Conductance calculation failed: {str(e)}")
        
        messagebox.showinfo("Community Metrics", "\n".join(metrics))
    
    def calculate_pagerank(self):
        if self.G is None:
            messagebox.showerror("Error", "No graph loaded")
            return
            
        try:
            pagerank = nx.pagerank(self.G)
            top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
            result = "PageRank Top 10:\n" + "\n".join([f"{node}: {score:.4f}" for node, score in top_nodes])
            messagebox.showinfo("PageRank Results", result)
        except Exception as e:
            messagebox.showerror("Error", f"PageRank calculation failed: {str(e)}")
    
    def calculate_betweenness(self):
        if self.G is None:
            messagebox.showerror("Error", "No graph loaded")
            return
            
        try:
            betweenness = nx.betweenness_centrality(self.G)
            top_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
            result = "Betweenness Centrality Top 10:\n" + "\n".join([f"{node}: {score:.4f}" for node, score in top_nodes])
            messagebox.showinfo("Betweenness Results", result)
        except Exception as e:
            messagebox.showerror("Error", f"Betweenness calculation failed: {str(e)}")
    
    def calculate_clustering(self):
        if self.G is None:
            messagebox.showerror("Error", "No graph loaded")
            return
            
        try:
            clustering = nx.clustering(self.G)
            avg_clustering = nx.average_clustering(self.G)
            transitivity = nx.transitivity(self.G)
            
            top_nodes = sorted(clustering.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Create visualization
            self.figure.clf()
            ax = self.figure.add_subplot(111)
            
            # Plot histogram of clustering coefficients
            ax.hist(clustering.values(), bins=20, color='lightgreen', edgecolor='black')
            ax.set_title("Clustering Coefficient Distribution")
            ax.set_xlabel("Clustering Coefficient")
            ax.set_ylabel("Count")
            
            # Display statistics
            stats = (
                f"Clustering Statistics:\n"
                f"Average clustering coefficient: {avg_clustering:.4f}\n"
                f"Global clustering (transitivity): {transitivity:.4f}\n\n"
                f"Top 10 Nodes by Clustering:\n"
            )
            stats += "\n".join([f"{node}: {coeff:.4f}" for node, coeff in top_nodes])
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, stats)
            
            self.canvas.draw()
            self.log_message("Calculated clustering coefficients")
            
        except Exception as e:
            messagebox.showerror("Error", f"Clustering calculation failed: {str(e)}")
    
    def plot_degree_distribution(self):
        if self.G is None:
            messagebox.showerror("Error", "No graph loaded")
            return
            
        try:
            degrees = [d for _, d in self.G.degree()]
            self.figure.clf()
            ax = self.figure.add_subplot(111)
            
            ax.hist(degrees, bins=20, color='skyblue', edgecolor='black')
            ax.set_title("Degree Distribution")
            ax.set_xlabel("Degree")
            ax.set_ylabel("Count")
            
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot degree distribution: {str(e)}")
    
    def calculate_path_length(self):
        if self.G is None:
            messagebox.showerror("Error", "No graph loaded")
            return
            
        try:
            if nx.is_connected(self.G):
                avg_path = nx.average_shortest_path_length(self.G)
                messagebox.showinfo("Path Length", f"Average Shortest Path Length: {avg_path:.3f}")
            else:
                messagebox.showinfo("Path Length", "Graph is not connected - showing largest component")
                largest_cc = max(nx.connected_components(self.G.to_undirected()), key=len)
                subgraph = self.G.subgraph(largest_cc)
                avg_path = nx.average_shortest_path_length(subgraph)
                messagebox.showinfo("Path Length", f"Average Path in Largest Component: {avg_path:.3f}")
        except Exception as e:
            messagebox.showerror("Error", f"Path length calculation failed: {str(e)}")
    
    def show_top_degrees(self):
        """Display top 10 nodes by degree centrality"""
        try:
            if self.G is None:
                messagebox.showerror("Error", "No graph loaded")
                return

            degrees = dict(self.G.degree())
            sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            
            result = "Top 10 Nodes by Degree:\n"
            result += "\n".join([f"{node}: {degree}" for node, degree in sorted_degrees])
            
            # Display in output text
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, result)
            
            messagebox.showinfo("Top Degrees", result)
            self.log_message("Displayed top 10 degrees")
            
        except Exception as e:
            self.show_error("Error showing degrees", str(e))

    def show_top_closeness(self):
        """Display top 10 nodes by closeness centrality"""
        try:
            if self.G is None:
                messagebox.showerror("Error", "No graph loaded")
                return

            closeness = nx.closeness_centrality(self.G)
            sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
            
            result = "Top 10 Nodes by Closeness Centrality:\n"
            result += "\n".join([f"{node}: {score:.4f}" for node, score in sorted_closeness])
            
            # Display in output text
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, result)
            
            messagebox.showinfo("Top Closeness", result)
            self.log_message("Displayed top 10 closeness centrality")
            
        except Exception as e:
            self.show_error("Error showing closeness", str(e))

    def filter_centrality(self, centrality_type):
        """Filter nodes based on centrality measure"""
        try:
            if self.G is None:
                messagebox.showerror("Error", "No graph loaded")
                return
                
            threshold = self.filter_threshold.get()
            
            if centrality_type == 'degree':
                centrality = dict(self.G.degree())
                title = f"Nodes with Degree ≥ {threshold}"
                color = '#4a6baf'
            elif centrality_type == 'betweenness':
                centrality = nx.betweenness_centrality(self.G)
                title = f"Nodes with Betweenness Centrality ≥ {threshold}"
                color = '#28a745'
            elif centrality_type == 'closeness':
                centrality = nx.closeness_centrality(self.G)
                title = f"Nodes with Closeness Centrality ≥ {threshold}"
                color = '#dc3545'
            elif centrality_type == 'harmonic':
                centrality = nx.harmonic_centrality(self.G)
                title = f"Nodes with Harmonic Centrality ≥ {threshold}"
                color = '#800080'
            else:
                raise ValueError("Invalid centrality type")
            
            filtered_nodes = [n for n in self.G.nodes() if centrality[n] >= threshold]
            filtered_G = self.G.subgraph(filtered_nodes)
            
            self.figure.clf()
            ax = self.figure.add_subplot(111)
            
            pos = self.calculate_layout()
            node_sizes = [float(self.size_scale.get()) * (1 + self.G.degree(n)) / 10 for n in filtered_nodes]
            
            nx.draw_networkx_nodes(
                filtered_G, pos,
                node_color=color,
                node_size=node_sizes,
                ax=ax
            )
            
            if self.directed_var.get():
                nx.draw_networkx_edges(
                    filtered_G, pos,
                    arrows=True,
                    ax=ax
                )
            else:
                nx.draw_networkx_edges(
                    filtered_G, pos,
                    ax=ax
                )
            
            if len(filtered_nodes) <= 50:
                nx.draw_networkx_labels(filtered_G, pos, font_size=8, ax=ax)
            
            plt.title(title, pad=20)
            plt.axis('off')
            
            self.canvas.draw()
            
            # Display centrality values in output
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"{title}\n\n")
            for node in sorted(filtered_nodes, key=lambda x: centrality[x], reverse=True):
                self.output_text.insert(tk.END, f"Node {node}: {centrality_type} = {centrality[node]:.1f}\n")
            
            self.log_message(f"Filtered by {centrality_type} (threshold={threshold})")
            
        except Exception as e:
            self.show_error("Error in centrality filtering", str(e))

    def show_conductance(self):
        """Calculate and display conductance values"""
        try:
            if self.G is None:
                messagebox.showerror("Error", "No graph loaded")
                return
                
            if not self.communities:
                messagebox.showerror("Error", "Run community detection first")
                return
                
            communities = defaultdict(list)
            for node, comm_id in self.communities.items():
                communities[comm_id].append(node)
            
            community_sets = [set(nodes) for nodes in communities.values()]
            
            conductances = []
            conductance_details = []
            for i, comm in enumerate(community_sets):
                if len(comm) == 0:
                    continue
                cut = nx.cut_size(self.G, comm)
                volume = nx.volume(self.G, comm)
                total = self.G.number_of_edges() * 2
                conductance = cut / min(volume, total - volume) if min(volume, total - volume) != 0 else 0
                conductances.append(conductance)
                conductance_details.append(f"Community {i}: {conductance:.4f}")
            
            avg_conductance = np.mean(conductances)
            message = "Conductance Values:\n\n" + "\n".join(conductance_details)
            message += f"\n\nAverage Conductance: {avg_conductance:.4f}"
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, message)
            messagebox.showinfo("Conductance Results", message)
            self.log_message("Conductance values calculated")
            
        except Exception as e:
            self.show_error("Error calculating conductance", str(e))

    def show_modularity(self):
        """Calculate and display modularity"""
        try:
            if self.G is None:
                messagebox.showerror("Error", "No graph loaded")
                return
                
            if not self.communities:
                messagebox.showerror("Error", "Run community detection first")
                return
                
            communities = defaultdict(list)
            for node, comm_id in self.communities.items():
                communities[comm_id].append(node)
            
            community_sets = [set(nodes) for nodes in communities.values()]
            
            mod = nx.community.modularity(self.G, community_sets)
            message = f"Modularity: {mod:.4f}\nNumber of communities: {len(communities)}"
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, message)
            messagebox.showinfo("Modularity Results", message)
            self.log_message(f"Modularity calculated: {mod:.4f}")
            
        except Exception as e:
            self.show_error("Error calculating modularity", str(e))

    def show_nmi(self):
        try:
            if self.G is None or not self.node_attributes or 'class' not in next(iter(self.node_attributes.values()), {}):
                raise ValueError("Node data with ground truth classes is required")
            
            if not self.communities:
                messagebox.showerror("Error", "Run community detection first")
                return

            ground_truth = {node: data.get('class', 0) for node, data in self.G.nodes(data=True)}
            truth = [ground_truth[node] for node in self.G.nodes()]

            # Handle communities as either dict or list of sets/lists
            if isinstance(self.communities, dict):
                detected = [self.communities[node] for node in self.G.nodes()]
            elif isinstance(self.communities, (list, tuple)):
                node_to_community = {}
                for i, community in enumerate(self.communities):
                    for node in community:
                        node_to_community[node] = i
                detected = [node_to_community[node] for node in self.G.nodes()]
            else:
                raise ValueError("Unrecognized format for communities")

            nmi = normalized_mutual_info_score(truth, detected)
            message = f"Normalized Mutual Information (NMI): {nmi:.4f}"

            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, message)
            messagebox.showinfo("NMI Results", message)
            self.log_message(f"NMI calculated: {nmi:.4f}")

        except Exception as e:
            self.show_error("Error calculating NMI", str(e))

    def show_ari(self):
        """Calculate and display Adjusted Rand Index"""
        try:
            if self.G is None or not self.node_attributes or 'class' not in next(iter(self.node_attributes.values()), {}):
                raise ValueError("Node data with ground truth classes is required")
                
            if not self.communities:
                messagebox.showerror("Error", "Run community detection first")
                return
                
            ground_truth = {node: data.get('class', 0) for node, data in self.G.nodes(data=True)}
            detected = [self.communities[node] for node in self.G.nodes()]
            truth = [ground_truth[node] for node in self.G.nodes()]
            
            ari = adjusted_rand_score(truth, detected)
            message = f"Adjusted Rand Index (ARI): {ari:.4f}"
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, message)
            messagebox.showinfo("ARI Results", message)
            self.log_message(f"ARI calculated: {ari:.4f}")
            
        except Exception as e:
            self.show_error("Error calculating ARI", str(e))

    def show_harmonic_centrality(self):
        """Calculate and display harmonic centrality for each node"""
        try:
            if self.G is None:
                messagebox.showerror("Error", "No graph loaded")
                return
                
            harmonic = nx.harmonic_centrality(self.G)
            sorted_nodes = sorted(harmonic.items(), key=lambda x: x[1], reverse=True)
            top_nodes = "\n".join([f"Node {node}: {centrality:.4f}" for node, centrality in sorted_nodes[:5]])
            avg_harmonic = sum(harmonic.values()) / len(harmonic)
            message = f"Top 5 Nodes:\n{top_nodes}\n\nAverage Harmonic Centrality: {avg_harmonic:.4f}"
            
            full_output = "Harmonic Centrality Scores:\n\n" + "\n".join(
                [f"Node {node}: {centrality:.4f}" for node, centrality in sorted_nodes]
            ) + f"\n\nAverage: {avg_harmonic:.4f}"
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, full_output)
            messagebox.showinfo("Harmonic Centrality Results", message)
            self.log_message("Harmonic centrality calculated")
            
        except Exception as e:
            self.show_error("Error calculating harmonic centrality", str(e))

   
    def visualize_metric_distribution(self, metric_name):
        """Create visualization for a specific metric distribution"""
        if self.G is None:
            self.show_error("Error", "No graph loaded")
            return
        
        try:
            # Get the appropriate metric data
            if metric_name == "degree":
                values = dict(self.G.degree()).values()
                title = "Degree Distribution"
                xlabel = "Degree"
            elif metric_name == "betweenness":
                values = nx.betweenness_centrality(self.G).values()
                title = "Betweenness Centrality Distribution"
                xlabel = "Betweenness Centrality"
            elif metric_name == "closeness":
                values = nx.closeness_centrality(self.G).values()
                title = "Closeness Centrality Distribution"
                xlabel = "Closeness Centrality"
            elif metric_name == "clustering":
                values = nx.clustering(self.G).values()
                title = "Clustering Coefficient Distribution"
                xlabel = "Clustering Coefficient"
            else:
                self.show_error("Error", "Invalid metric name")
                return
            
            # Create visualization
            self.figure.clf()
            ax = self.figure.add_subplot(111)
            
            # Plot histogram
            ax.hist(values, bins=20, color='skyblue', edgecolor='black')
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            
            # Calculate and display statistics
            values_list = list(values)
            stats = (
                f"{title} Statistics:\n"
                f"Mean: {np.mean(values_list):.4f}\n"
                f"Median: {np.median(values_list):.4f}\n"
                f"Standard deviation: {np.std(values_list):.4f}\n"
                f"Minimum: {min(values_list):.4f}\n"
                f"Maximum: {max(values_list):.4f}"
            )
            
            # Add stats to plot
            ax.text(0.95, 0.95, stats, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            self.canvas.draw()
            
            # Also show in output
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, stats)
            self.log_message(f"Displayed {metric_name} distribution")
            
        except Exception as e:
            self.show_error("Visualization Error", str(e))

    def run_louvain(self):
        """Run Louvain algorithm and return partition"""
        try:
            return best_partition(self.G.to_undirected())
        except Exception as e:
            self.show_error("Louvain Error", str(e))
            return {}

    def run_girvan_newman(self):
        """Run Girvan-Newman algorithm and return first partition"""
        try:
            communities = nx.community.girvan_newman(self.G.to_undirected())
            first_partition = next(communities)
            return {node: i for i, comm in enumerate(first_partition) for node in comm}
        except StopIteration:
            self.show_error("Girvan-Newman Error", "No partitions generated")
            return {}
        except Exception as e:
            self.show_error("Girvan-Newman Error", str(e))
            return {}

    def compare_community_algorithms(self):
        """Main comparison function"""
        if self.G is None:
            self.show_error("Error", "Load a graph first")
            return
        
        try:
            # Create undirected copy for community detection
            undirected_graph = self.G.to_undirected()
            
            # Get partitions
            louvain_part = self.run_louvain()
            girvan_part = self.run_girvan_newman()
            
            if not louvain_part or not girvan_part:
                return
            
            # Calculate metrics using undirected graph copy
            l_metrics = self.compute_community_metrics(louvain_part, undirected_graph)
            g_metrics = self.compute_community_metrics(girvan_part, undirected_graph)

            # Create comparison window
            top = tk.Toplevel(self.root)
            top.title("Algorithm Comparison")
            top.geometry("680x300")
            
            # Main frame with grid layout
            main_frame = ttk.Frame(top, padding=15)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            # Configure grid weights
            main_frame.grid_columnconfigure(1, weight=1)
            main_frame.grid_columnconfigure(2, weight=1)
            
            # Header
            ttk.Label(main_frame, text="Metric", font=('Arial', 10, 'bold')).grid(
                row=0, column=0, sticky='w', padx=5, pady=2)
            ttk.Label(main_frame, text="Louvain", font=('Arial', 10, 'bold')).grid(
                row=0, column=1, sticky='w', padx=5, pady=2)
            ttk.Label(main_frame, text="Girvan-Newman", font=('Arial', 10, 'bold')).grid(
                row=0, column=2, sticky='w', padx=5, pady=2)

            # Metrics to display
            metrics = [
                ('Number of Communities', 'num_communities', 'd'),
                ('Modularity Score', 'modularity', '.3f'),
                ('Average Conductance', 'conductance', '.3f'),
                ('Normalized MI', 'nmi', '.3f'),
                ('Adjusted RI', 'ari', '.3f')
            ]

            # Populate metrics
            for row, (label, key, fmt) in enumerate(metrics, 1):
                ttk.Label(main_frame, text=label).grid(
                    row=row, column=0, sticky='w', padx=5, pady=2)
                
                # Louvain value
                l_val = l_metrics.get(key)
                l_text = f"{l_val:{fmt}}" if l_val is not None else "N/A"
                ttk.Label(main_frame, text=l_text).grid(
                    row=row, column=1, sticky='w', padx=5, pady=2)
                
                # Girvan value
                g_val = g_metrics.get(key)
                g_text = f"{g_val:{fmt}}" if g_val is not None else "N/A"
                ttk.Label(main_frame, text=g_text).grid(
                    row=row, column=2, sticky='w', padx=5, pady=2)

            # Close button
            close_btn = ttk.Button(top, text="Close", command=top.destroy)
            close_btn.pack(pady=10)

            # Bring window to front
            top.lift()
            top.attributes('-topmost', True)
            top.after_idle(top.attributes, '-topmost', False)

        except Exception as e:
            self.show_error("Comparison Error", f"Failed to create comparison: {str(e)}")
            print(traceback.format_exc())

    def compute_community_metrics(self, partition, undirected_graph):
        """Calculate metrics for a given partition using undirected graph"""
        metrics = {}
        communities = defaultdict(list)
        
        try:
            for node, comm_id in partition.items():
                communities[comm_id].append(node)
            
            community_sets = [set(nodes) for nodes in communities.values()]
            metrics['num_communities'] = len(communities)

            # Modularity calculation
            try:
                metrics['modularity'] = nx.community.modularity(
                    undirected_graph, community_sets)
            except Exception as e:
                metrics['modularity'] = None

            # Conductance calculation
            conductances = []
            for comm in community_sets:
                try:
                    if len(comm) == 0 or len(comm) == undirected_graph.number_of_nodes():
                        continue
                    
                    cut = nx.cut_size(undirected_graph, comm)
                    volume = nx.volume(undirected_graph, comm)
                    total = 2 * undirected_graph.number_of_edges()  # For undirected
                    
                    if volume == 0 or (total - volume) == 0:
                        continue
                        
                    conductance = cut / min(volume, total - volume)
                    conductances.append(conductance)
                except:
                    continue
            
            metrics['conductance'] = np.mean(conductances) if conductances else None

            # NMI and ARI calculations
            metrics['nmi'] = None
            metrics['ari'] = None
            if hasattr(self, 'node_attributes') and 'class' in next(iter(self.node_attributes.values()), {}):
                try:
                    ground_truth = {node: data.get('class', 0) 
                                   for node, data in undirected_graph.nodes(data=True)}
                    truth = [ground_truth[node] for node in undirected_graph.nodes()]
                    detected = [partition[node] for node in undirected_graph.nodes()]
                    metrics['nmi'] = normalized_mutual_info_score(truth, detected)
                    metrics['ari'] = adjusted_rand_score(truth, detected)
                except Exception as e:
                    pass

        except Exception as e:
            self.show_error("Metric Calculation Error", str(e))
            print(traceback.format_exc())

        return metrics
        

    def clear_all(self):
        """Clear all data and visualizations"""
        self.figure.clear()
        self.canvas.draw()
        self.output_text.delete(1.0, tk.END)
        self.node_attributes = {}
        self.edge_attributes = pd.DataFrame()
        self.G = None
        self.communities = {}
        self.log_message("All data cleared")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SocialNetworkAnalyzer(root)
        root.mainloop()
    except Exception as e:
        error_msg = f"Unhandled exception: {str(e)}\n{traceback.format_exc()}"
        messagebox.showerror("Critical Error", error_msg)
        sys.exit(1)