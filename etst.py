import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from community import best_partition
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
from collections import defaultdict
import math
import traceback
import sys

class SocialNetworkAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Social Network Analysis Tool")
        self.root.geometry("1400x900")
        
        # Initialize data structures
        self.G = None
        self.original_graph = None
        self.node_attributes = {}
        self.edge_attributes = pd.DataFrame()
        self.communities = {}
        self.layout_pos = None
        self.current_layout = "spring"
        self.layout_seed = 42
        
        # Visualization parameters
        self.node_size = 300
        self.custom_color = "#1f78b4"
        self.node_shape = "o"
        self.edge_width = 1.0
        self.edge_style = "solid"
        self.arrow_size = 10
        self.color_var = tk.StringVar(value="manual")
        
        # Initialize UI
        self.setup_ui()
        self.setup_styles()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', font=('Arial', 10), padding=5)
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'))

    def setup_ui(self):
        self.create_control_panel()
        self.create_visualization_area()
        self.create_output_console()

    def create_control_panel(self):
        control_frame = ttk.Frame(self.root, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)

        # Create scrollable container
        canvas = tk.Canvas(control_frame)
        scrollbar = ttk.Scrollbar(control_frame, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add control sections
        self.create_data_loading_section(scroll_frame)
        self.create_visualization_controls(scroll_frame)
        self.create_analysis_controls(scroll_frame)
        self.create_filter_controls(scroll_frame)
        self.create_metrics_controls(scroll_frame)

    def create_visualization_area(self):
        self.figure = plt.figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def create_output_console(self):
        output_frame = ttk.Frame(self.root)
        output_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        
        self.output_text = tk.Text(output_frame, wrap=tk.WORD, font=('Consolas', 10))
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(self.output_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.output_text.yview)

    def create_data_loading_section(self, parent):
        frame = ttk.LabelFrame(parent, text="Data Loading", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(frame, text="Load Nodes CSV", command=self.load_nodes).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Load Edges CSV", command=self.load_edges).pack(fill=tk.X, pady=2)
        
        self.directed_var = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Directed Graph", variable=self.directed_var).pack(anchor=tk.W)

    def create_visualization_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Visualization Settings", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        # Layout selection
        ttk.Label(frame, text="Layout:").pack(anchor=tk.W)
        self.layout_var = tk.StringVar(value="spring")
        ttk.Combobox(frame, textvariable=self.layout_var, 
                    values=["spring", "circular", "kamada-kawai", "spectral", "shell", "tree", "radial"]).pack(fill=tk.X)
        
        # Node controls
        ttk.Label(frame, text="Node Size:").pack(anchor=tk.W)
        self.size_scale = ttk.Scale(frame, from_=10, to=1000, value=300)
        self.size_scale.pack(fill=tk.X)
        
        ttk.Label(frame, text="Node Color:").pack(anchor=tk.W)
        color_frame = ttk.Frame(frame)
        color_frame.pack(fill=tk.X)
        ttk.Button(color_frame, text="Choose Color", command=self.choose_node_color).pack(side=tk.LEFT)
        self.color_preview = tk.Canvas(color_frame, width=50, height=20, bg=self.custom_color)
        self.color_preview.pack(side=tk.LEFT, padx=5)
        
        ttk.Combobox(frame, textvariable=self.color_var, 
                    values=["manual", "community", "degree", "in_degree", "out_degree", "betweenness", "pagerank", "eigenvector"]).pack(fill=tk.X)
        
        # Edge controls
        ttk.Label(frame, text="Edge Style:").pack(anchor=tk.W)
        self.edge_style_var = tk.StringVar(value="solid")
        ttk.Combobox(frame, textvariable=self.edge_style_var,
                    values=["solid", "dashed", "dotted", "dashdot"]).pack(fill=tk.X)
        
        ttk.Label(frame, text="Edge Width:").pack(anchor=tk.W)
        self.edge_width_scale = ttk.Scale(frame, from_=1, to=10, value=1)
        self.edge_width_scale.pack(fill=tk.X)
        
        ttk.Button(frame, text="Update Visualization", command=self.update_visualization).pack(fill=tk.X, pady=5)
        ttk.Button(frame, text="Reapply Layout", command=self.force_new_layout).pack(fill=tk.X)

    def create_analysis_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Analysis Tools", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        buttons = [
            ("Run Louvain Community Detection", lambda: self.run_community_detection("louvain")),
            ("Run Girvan-Newman Community Detection", lambda: self.run_community_detection("girvan")),
            ("Compare Community Algorithms", self.compare_community_algorithms),
            ("Calculate PageRank", self.calculate_pagerank),
            ("Calculate Betweenness Centrality", self.calculate_betweenness),
            ("Calculate Clustering Coefficients", self.calculate_clustering),
            ("Calculate Eigenvector Centrality", self.calculate_eigenvector),
            ("Show Degree Distribution", self.plot_degree_distribution),
            ("Calculate Average Path Length", self.calculate_path_length),
            ("Show Top 10 Degrees", self.show_top_degrees),
            ("Show Top 10 Closeness", self.show_top_closeness)
        ]
        
        for text, command in buttons:
            ttk.Button(frame, text=text, command=command).pack(fill=tk.X, pady=2)

    def create_filter_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Centrality Filters", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        self.filter_threshold = tk.DoubleVar(value=0.0)
        ttk.Entry(frame, textvariable=self.filter_threshold).pack(fill=tk.X)
        
        filters = [
            ("Filter by Degree", 'degree'),
            ("Filter by In-Degree", 'in_degree'),
            ("Filter by Out-Degree", 'out_degree'),
            ("Filter by Betweenness", 'betweenness'),
            ("Filter by Closeness", 'closeness'),
            ("Filter by Harmonic", 'harmonic'),
            ("Filter by Eigenvector", 'eigenvector')
        ]
        
        for text, centrality_type in filters:
            ttk.Button(frame, text=text, command=lambda ct=centrality_type: self.filter_centrality(ct)).pack(fill=tk.X, pady=2)

    def create_metrics_controls(self, parent):
        frame = ttk.LabelFrame(parent, text="Metrics", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        metrics = [
            ("Show Conductance", self.show_conductance),
            ("Show Modularity", self.show_modularity),
            ("Show NMI", self.show_nmi),
            ("Show Harmonic Centrality", self.show_harmonic_centrality),
            ("Clear All", self.clear_all)
        ]
        
        for text, command in metrics:
            ttk.Button(frame, text=text, command=command).pack(fill=tk.X, pady=2)

    def log_message(self, message):
        self.output_text.insert(tk.END, f"[LOG] {message}\n")
        self.output_text.see(tk.END)

    def handle_error(self, title, message):
        self.log_message(f"{title}: {message}")
        messagebox.showerror(title, message)

    def load_nodes(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path: return
        
        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.lower()
            
            if 'id' not in df.columns:
                self.handle_error("Error", "Node CSV must contain 'id' column")
                return
                
            self.node_attributes = df.set_index('id').to_dict('index')
            self.log_message(f"Loaded {len(self.node_attributes)} nodes")
            self.layout_pos = None
            
        except Exception as e:
            self.handle_error("Node Loading Error", str(e))

    def load_edges(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path: return
        
        try:
            self.edge_attributes = pd.read_csv(file_path)
            self.edge_attributes.columns = self.edge_attributes.columns.str.lower()
            
            if not {'source', 'target'}.issubset(self.edge_attributes.columns):
                self.handle_error("Error", "Edge CSV must contain 'source' and 'target'")
                return
                
            self.create_graph()
            self.update_visualization()
            self.log_message(f"Loaded {len(self.edge_attributes)} edges")
            
        except Exception as e:
            self.handle_error("Edge Loading Error", str(e))

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
            self.handle_error("Graph Creation Error", str(e))

    def update_visualization(self):
        if not self.G or len(self.G.nodes) == 0: return
        
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        
        if not self.layout_pos or self.current_layout != self.layout_var.get():
            self.current_layout = self.layout_var.get()
            self.layout_pos = self.calculate_layout()
        
        node_colors = self.get_node_colors()
        node_size = float(self.size_scale.get())
        edge_style = self.edge_style_var.get()
        edge_width = float(self.edge_width_scale.get())
        
        nx.draw_networkx_nodes(
            self.G, self.layout_pos, ax=ax,
            node_size=node_size,
            node_color=node_colors,
            cmap=plt.cm.viridis if self.color_var.get() != "manual" else None,
            node_shape=self.node_shape
        )
        
        nx.draw_networkx_edges(
            self.G, self.layout_pos, ax=ax,
            width=edge_width,
            style=edge_style,
            arrows=self.G.is_directed(),
            arrowsize=10
        )
        
        if len(self.G.nodes) <= 50:
            nx.draw_networkx_labels(self.G, self.layout_pos, ax=ax, font_size=8)
        
        self.canvas.draw()

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
            return nx.spring_layout(self.G, seed=self.layout_seed)
        except Exception as e:
            self.handle_error("Layout Error", str(e))
            return nx.spring_layout(self.G, seed=self.layout_seed)

    def tree_layout(self):
        if not self.G: return {}
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
        if not self.G: return {}
        root = max(self.G.degree(), key=lambda x: x[1])[0]
        levels = nx.single_source_shortest_path_length(self.G, root)
        max_level = max(levels.values()) if levels else 1
        
        pos = {}
        for node, level in levels.items():
            angle = (hash(node) % 628) / 100
            radius = level / max_level
            pos[node] = (radius * math.cos(angle), radius * math.sin(angle))
        return pos

    def get_node_colors(self):
        color_by = self.color_var.get()
        if color_by == "manual":
            return [self.custom_color] * len(self.G.nodes)  
        
        if color_by == "community":
            return [self.communities.get(node, 0) for node in self.G.nodes]
        
        metrics = {}
        if self.G.is_directed():
            metrics.update({
                'in_degree': dict(self.G.in_degree()),
                'out_degree': dict(self.G.out_degree()),
            })
        metrics.update({
            'degree': dict(self.G.degree()),
            'betweenness': nx.betweenness_centrality(self.G),
            'pagerank': nx.pagerank(self.G),
            'eigenvector': nx.eigenvector_centrality(self.G)
        })
        return [metrics.get(color_by, {}).get(node, 0) for node in self.G.nodes]

    def run_community_detection(self, method):
        try:
            if method == "louvain":
                self.communities = best_partition(self.G.to_undirected())
                message = "Louvain communities detected"
            elif method == "girvan":
                communities = nx.community.girvan_newman(self.G.to_undirected())
                self.communities = {node: i for i, comm in enumerate(next(communities)) for node in comm}
                message = "Girvan-Newman communities detected"
            
            self.color_var.set("community")
            self.update_visualization()
            messagebox.showinfo("Success", message)
            
        except Exception as e:
            self.handle_error("Community Detection Error", str(e))

    def compare_community_algorithms(self):
        try:
            louvain_part = best_partition(self.G.to_undirected())
            girvan_part = self.run_girvan_newman()
            
            l_metrics = self.compute_metrics(louvain_part)
            g_metrics = self.compute_metrics(girvan_part)
            
            top = tk.Toplevel()
            top.title("Algorithm Comparison")
            
            metrics = [
                ("Number of Communities", "num_communities", "d"),
                ("Modularity Score", "modularity", ".3f"),
                ("Average Conductance", "conductance", ".3f"),
                ("Normalized MI", "nmi", ".3f")
            ]
            
            for i, (label, key, fmt) in enumerate(metrics):
                ttk.Label(top, text=label).grid(row=i, column=0, sticky='w')
                ttk.Label(top, text=f"{l_metrics.get(key, 0):{fmt}}").grid(row=i, column=1)
                ttk.Label(top, text=f"{g_metrics.get(key, 0):{fmt}}").grid(row=i, column=2)
            
            ttk.Button(top, text="Close", command=top.destroy).grid(row=len(metrics)+1, columnspan=3)
            
        except Exception as e:
            self.handle_error("Comparison Error", str(e))

    def compute_metrics(self, partition):
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        community_sets = [set(nodes) for nodes in communities.values()]
        metrics = {
            'num_communities': len(communities),
            'modularity': nx.community.modularity(self.G, community_sets)
        }
        
        conductances = []
        for comm in community_sets:
            cut = nx.cut_size(self.G, comm)
            volume = nx.volume(self.G, comm)
            total = self.G.number_of_edges() * 2
            if min(volume, total - volume) != 0:
                conductances.append(cut / min(volume, total - volume))
        metrics['conductance'] = np.mean(conductances) if conductances else 0
        
        if 'class' in next(iter(self.node_attributes.values()), {}):
            ground_truth = [data.get('class', 0) for _, data in self.G.nodes(data=True)]
            detected = [partition[node] for node in self.G.nodes]
            metrics['nmi'] = normalized_mutual_info_score(ground_truth, detected)
            
        return metrics

    def calculate_pagerank(self):
        try:
            pagerank = nx.pagerank(self.G)
            top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
            result = "PageRank Top 10:\n" + "\n".join([f"{node}: {score:.4f}" for node, score in top_nodes])
            messagebox.showinfo("PageRank Results", result)
        except Exception as e:
            self.handle_error("PageRank Error", str(e))

    def calculate_betweenness(self):
        try:
            betweenness = nx.betweenness_centrality(self.G)
            top_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:10]
            result = "Betweenness Centrality Top 10:\n" + "\n".join([f"{node}: {score:.4f}" for node, score in top_nodes])
            messagebox.showinfo("Betweenness Results", result)
        except Exception as e:
            self.handle_error("Betweenness Error", str(e))

    def calculate_clustering(self):
        if self.G is None:
            messagebox.showerror("Error", "No graph loaded")
            return
            
        try:
            clustering = nx.clustering(self.G)
            avg_clustering = nx.average_clustering(self.G)
            result = f"Average Clustering Coefficient: {avg_clustering:.3f}\n\n"
            result += "Top 10 Nodes by Clustering:\n"
            top_nodes = sorted(clustering.items(), key=lambda x: x[1], reverse=True)[:10]
            result += "\n".join([f"{node}: {score:.3f}" for node, score in top_nodes])
            messagebox.showinfo("Clustering Results", result)
        except Exception as e:
            messagebox.showerror("Error", f"Clustering calculation failed: {str(e)}")
    def calculate_eigenvector(self):
     try:
        if self.G.is_directed() and not nx.is_strongly_connected(self.G):
            raise ValueError("Graph must be strongly connected for eigenvector centrality in directed graphs")
        
        eigenvector = nx.eigenvector_centrality(self.G)
        top_nodes = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Create visualization
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.hist(eigenvector.values(), bins=20, color='lightblue', edgecolor='black')
        ax.set_title("Eigenvector Centrality Distribution")
        self.canvas.draw()
        
        # Create message for messagebox
        result = "Top 10 Eigenvector Centrality Scores:\n"
        result += "\n".join([f"{node}: {score:.4f}" for node, score in top_nodes])
        
        # Show in messagebox
        messagebox.showinfo("Eigenvector Centrality Results", result)
        
     except Exception as e:
        self.handle_error("Eigenvector Error", str(e))

    def plot_degree_distribution(self):
        try:
            degrees = [d for _, d in self.G.degree()]
            self.figure.clf()
            ax = self.figure.add_subplot(111)
            ax.hist(degrees, bins=20, color='skyblue', edgecolor='black')
            ax.set_title("Degree Distribution")
            self.canvas.draw()
        except Exception as e:
            self.handle_error("Degree Distribution Error", str(e))

    def calculate_path_length(self):
        try:
            if nx.is_connected(self.G):
                avg_path = nx.average_shortest_path_length(self.G)
                messagebox.showinfo("Path Length", f"Average Path Length: {avg_path:.3f}")
            else:
                largest_cc = max(nx.connected_components(self.G.to_undirected()), key=len)
                subgraph = self.G.subgraph(largest_cc)
                avg_path = nx.average_shortest_path_length(subgraph)
                messagebox.showinfo("Path Length", f"Average in Largest Component: {avg_path:.3f}")
        except Exception as e:
            self.handle_error("Path Length Error", str(e))

    def show_top_degrees(self):
        try:
            degrees = dict(self.G.degree())
            sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
            result = "Top 10 Degrees:\n" + "\n".join([f"{node}: {degree}" for node, degree in sorted_degrees])
            messagebox.showinfo("Top Degrees", result)
        except Exception as e:
            self.handle_error("Degree Error", str(e))

    def show_top_closeness(self):
        try:
            closeness = nx.closeness_centrality(self.G)
            sorted_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:10]
            result = "Top 10 Closeness:\n" + "\n".join([f"{node}: {score:.4f}" for node, score in sorted_closeness])
            messagebox.showinfo("Top Closeness", result)
        except Exception as e:
            self.handle_error("Closeness Error", str(e))

    def filter_centrality(self, centrality_type):
        try:
           
            if not self.G.is_directed() and centrality_type in ['in_degree', 'out_degree']:
                centrality_type = 'degree'
            
            metrics = {}
            if self.G.is_directed():
                metrics.update({
                    'in_degree': dict(self.G.in_degree()),
                    'out_degree': dict(self.G.out_degree()),
                })
            metrics.update({
                'degree': dict(self.G.degree()),
                'betweenness': nx.betweenness_centrality(self.G),
                'closeness': nx.closeness_centrality(self.G),
                'harmonic': nx.harmonic_centrality(self.G),
                'eigenvector': nx.eigenvector_centrality(self.G)
            })
            
            threshold = self.filter_threshold.get()
            filtered_nodes = [n for n in self.G.nodes if metrics[centrality_type][n] >= threshold]
            filtered_G = self.G.subgraph(filtered_nodes)
            
            self.figure.clf()
            ax = self.figure.add_subplot(111)
            pos = self.layout_pos or self.calculate_layout()
            node_sizes = [float(self.size_scale.get()) * (1 + self.G.degree(n)) / 10 for n in filtered_nodes]
            
            nx.draw_networkx_nodes(filtered_G, pos, node_size=node_sizes, ax=ax)
            nx.draw_networkx_edges(filtered_G, pos, ax=ax, arrows=self.G.is_directed())
            
            if len(filtered_nodes) <= 50:
                nx.draw_networkx_labels(filtered_G, pos, font_size=8, ax=ax)
            
            plt.title(f"Nodes with {centrality_type.capitalize()} ≥ {threshold}", pad=20)
            self.canvas.draw()
            
        except Exception as e:
            self.handle_error("Filter Error", str(e))

    def show_conductance(self):
        try:
            community_sets = [set(nodes) for nodes in self.get_communities().values()]
            conductances = []
            for comm in community_sets:
                cut = nx.cut_size(self.G, comm)
                volume = nx.volume(self.G, comm)
                total = self.G.number_of_edges() * 2
                if min(volume, total - volume) != 0:
                    conductances.append(cut / min(volume, total - volume))
            
            result = f"Average Conductance: {np.mean(conductances):.3f}"
            messagebox.showinfo("Conductance", result)
        except Exception as e:
            self.handle_error("Conductance Error", str(e))

    def show_modularity(self):
        try:
            modularity = nx.community.modularity(self.G, self.get_communities().values())
            messagebox.showinfo("Modularity", f"Modularity Score: {modularity:.3f}")
        except Exception as e:
            self.handle_error("Modularity Error", str(e))

    def show_nmi(self):
        try:
            ground_truth = [data.get('class', 0) for _, data in self.G.nodes(data=True)]
            detected = [self.communities[node] for node in self.G.nodes]
            nmi = normalized_mutual_info_score(ground_truth, detected)
            messagebox.showinfo("NMI", f"Normalized Mutual Information: {nmi:.3f}")
        except Exception as e:
            self.handle_error("NMI Error", str(e))

    def show_harmonic_centrality(self):
        try:
            harmonic = nx.harmonic_centrality(self.G)
            sorted_nodes = sorted(harmonic.items(), key=lambda x: x[1], reverse=True)[:10]
            result = "Top 10 Harmonic Centrality:\n" + "\n".join([f"{node}: {score:.4f}" for node, score in sorted_nodes])
            messagebox.showinfo("Harmonic Centrality", result)
        except Exception as e:
            self.handle_error("Harmonic Error", str(e))

    def clear_all(self):
        self.figure.clf()
        self.canvas.draw()
        self.output_text.delete(1.0, tk.END)
        self.node_attributes = {}
        self.edge_attributes = pd.DataFrame()
        self.G = None
        self.communities = {}
        self.log_message("All data cleared")

    def get_communities(self):
        communities = defaultdict(list)
        for node, comm_id in self.communities.items():
            communities[comm_id].append(node)
        return communities

    def force_new_layout(self):
        self.layout_pos = None
        self.update_visualization()

    def choose_node_color(self):
        color = colorchooser.askcolor(initialcolor=self.custom_color)
        if color[1]:
            self.custom_color = color[1]
            self.color_preview.config(bg=self.custom_color)
            self.update_visualization()  # Force immediate update

    def run_girvan_newman(self):
        communities = nx.community.girvan_newman(self.G.to_undirected())
        return {node: i for i, comm in enumerate(next(communities)) for node in comm}

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SocialNetworkAnalyzer(root)
        root.mainloop()
    except Exception as e:
        error_msg = f"Critical Error: {str(e)}\n{traceback.format_exc()}"
        messagebox.showerror("Fatal Error", error_msg)
        sys.exit(1)