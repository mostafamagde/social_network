import pandas as pd
import networkx as nx
import community as community_louvain
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, colorchooser
from matplotlib.figure import Figure
import traceback
import sys
import random
from networkx.algorithms.community import girvan_newman

class NetworkAnalysisGUI:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Network Analysis Tool")
        master.configure(bg="#f0f0f0")
        
        # Initialize variables
        self.edge_df = None
        self.node_df = None
        self.G = None
        self.node_size = tk.IntVar(value=300)
        self.random_seed = tk.IntVar(value=42)
        self.selected_layout = tk.StringVar(value="spring")
        self.selected_community = tk.StringVar(value="louvain")
        
        # Node styling variables
        self.node_single_color = tk.StringVar(value="#4a6baf")
        self.node_shape = tk.StringVar(value="circle")
        self.node_label_visible = tk.BooleanVar(value=True)
        self.node_label_size = tk.IntVar(value=8)
        
        # Edge styling variables
        self.edge_single_color = tk.StringVar(value="#666666")
        self.edge_width = tk.IntVar(value=1)
        self.edge_style = tk.StringVar(value="solid")
        self.arrow_size = tk.IntVar(value=10)
        
        # Configure styles
        self.configure_styles()
        
        # Create main frames
        self.create_main_frames()
        
        # Create widgets
        self.create_widgets()
        
        # Center the window
        self.center_window()

    def configure_styles(self):
        """Configure custom styles for widgets"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Button styles
        style.configure('Primary.TButton', 
                      background='#4a6baf', 
                      foreground='white',
                      font=('Arial', 10, 'bold'),
                      padding=5,
                      borderwidth=2)
        
        style.configure('Secondary.TButton', 
                      background='#6c757d', 
                      foreground='white',
                      font=('Arial', 10, 'bold'),
                      padding=5,
                      borderwidth=2)
        
        style.configure('Danger.TButton', 
                      background='#dc3545', 
                      foreground='white',
                      font=('Arial', 10, 'bold'),
                      padding=5,
                      borderwidth=2)
        
        style.configure('Success.TButton', 
                      background='#28a745', 
                      foreground='white',
                      font=('Arial', 10, 'bold'),
                      padding=5,
                      borderwidth=2)
        
        # Label styles
        style.configure('Header.TLabel', 
                      background='#343a40', 
                      foreground='white',
                      font=('Arial', 12, 'bold'),
                      padding=5)
        
        # Entry styles
        style.configure('TEntry', 
                      font=('Arial', 10),
                      padding=5)

    def create_main_frames(self):
        """Create the main frames for the GUI"""
        # Main container frame
        self.main_container = ttk.Frame(self.master)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls (now with scrollbar)
        self.control_frame_container = ttk.Frame(self.main_container, width=300)
        self.control_frame_container.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create canvas and scrollbar for control frame
        self.control_canvas = tk.Canvas(self.control_frame_container)
        self.control_scrollbar = ttk.Scrollbar(self.control_frame_container, orient="vertical", command=self.control_canvas.yview)
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
        
        # Right panel - Output and Visualization
        self.right_panel = ttk.Frame(self.main_container)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Visualization frame
        self.visualization_frame = ttk.Frame(self.right_panel)
        self.visualization_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Output frame
        self.output_frame = ttk.Frame(self.right_panel)
        self.output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def create_widgets(self):
        """Create all widgets for the GUI"""
        self.create_control_widgets()
        self.create_output_widgets()
        self.create_visualization_widgets()

    def create_control_widgets(self):
        """Create widgets in the control frame"""
        # Data Loading Section
        load_frame = ttk.LabelFrame(self.scrollable_frame, text="Data Loading", style='Header.TLabel')
        load_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(load_frame, text="Load Edge CSV", style='Primary.TButton',
                  command=self.load_edge_file).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(load_frame, text="Load Node CSV", style='Primary.TButton',
                  command=self.load_node_file).pack(fill=tk.X, padx=5, pady=2)
        
        # Graph Type Selection
        self.graph_type = tk.StringVar(value='Undirected Graph')
        graph_frame = ttk.LabelFrame(self.scrollable_frame, text="Graph Type", style='Header.TLabel')
        graph_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(graph_frame, text="Undirected Graph", variable=self.graph_type,
                       value='Undirected Graph').pack(anchor=tk.W, padx=5, pady=2)
        
        ttk.Radiobutton(graph_frame, text="Directed Graph", variable=self.graph_type,
                       value='Directed Graph').pack(anchor=tk.W, padx=5, pady=2)
        
        # Visualization Settings
        vis_frame = ttk.LabelFrame(self.scrollable_frame, text="Visualization Settings", style='Header.TLabel')
        vis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Node size control
        ttk.Label(vis_frame, text="Node Size:").pack(anchor=tk.W, padx=5)
        ttk.Scale(vis_frame, from_=10, to=1000, variable=self.node_size,
                 command=lambda x: self.redraw_graph()).pack(fill=tk.X, padx=5, pady=2)
        
        # Layout algorithm selection
        ttk.Label(vis_frame, text="Layout Algorithm:").pack(anchor=tk.W, padx=5)
        layout_menu = ttk.OptionMenu(vis_frame, self.selected_layout, "spring", 
                                   "spring", "circular", "random", "shell", "spectral",
                                   "tree", "radial",
                                   command=lambda _: self.redraw_graph())
        layout_menu.pack(fill=tk.X, padx=5, pady=2)
        
        # Random seed control
        ttk.Label(vis_frame, text="Random Seed:").pack(anchor=tk.W, padx=5)
        ttk.Entry(vis_frame, textvariable=self.random_seed).pack(fill=tk.X, padx=5, pady=2)
        
        # Node Styling Section
        node_style_frame = ttk.LabelFrame(self.scrollable_frame, text="Node Styling", style='Header.TLabel')
        node_style_frame.pack(fill=tk.X, padx=5, pady=5)

        # Color Picker
        self.node_color_picker_frame = ttk.Frame(node_style_frame)
        ttk.Label(self.node_color_picker_frame, text="Node Color:").pack(side=tk.LEFT, padx=5)
        ttk.Button(self.node_color_picker_frame, text="Pick", 
                  command=self.pick_node_color).pack(side=tk.LEFT, padx=5)
        self.node_color_preview = tk.Canvas(self.node_color_picker_frame, width=20, height=20)
        self.node_color_preview.create_rectangle(0, 0, 20, 20, fill=self.node_single_color.get(), outline="")
        self.node_color_preview.pack(side=tk.LEFT)
        self.node_color_picker_frame.pack(fill=tk.X, padx=5, pady=2)

        # Node Shape
        ttk.Label(node_style_frame, text="Shape:").pack(anchor=tk.W, padx=5)
        shape_menu = ttk.OptionMenu(node_style_frame, self.node_shape, "circle", 
                                  "circle", "square", "triangle", "diamond",
                                  command=lambda _: self.redraw_graph())
        shape_menu.pack(fill=tk.X, padx=5, pady=2)

        # Node Labels
        ttk.Checkbutton(node_style_frame, text="Show Labels", 
                       variable=self.node_label_visible,
                       command=self.redraw_graph).pack(anchor=tk.W, padx=5)
        ttk.Label(node_style_frame, text="Label Size:").pack(anchor=tk.W, padx=5)
        ttk.Scale(node_style_frame, from_=6, to=20, variable=self.node_label_size,
                 command=lambda _: self.redraw_graph()).pack(fill=tk.X, padx=5, pady=2)
        
        # Edge Styling Section
        edge_style_frame = ttk.LabelFrame(self.scrollable_frame, text="Edge Styling", style='Header.TLabel')
        edge_style_frame.pack(fill=tk.X, padx=5, pady=5)

        # Edge Color
        ttk.Label(edge_style_frame, text="Color:").pack(anchor=tk.W, padx=5)
        ttk.Button(edge_style_frame, text="Pick Edge Color", 
                  command=self.pick_edge_color).pack(fill=tk.X, padx=5, pady=2)

        # Edge Width
        ttk.Label(edge_style_frame, text="Width:").pack(anchor=tk.W, padx=5)
        ttk.Scale(edge_style_frame, from_=1, to=10, variable=self.edge_width,
                 command=lambda _: self.redraw_graph()).pack(fill=tk.X, padx=5, pady=2)

        # Edge Style
        ttk.Label(edge_style_frame, text="Style:").pack(anchor=tk.W, padx=5)
        style_menu = ttk.OptionMenu(edge_style_frame, self.edge_style, "solid",
                                   "solid", "dashed", "dotted", "dashdot",
                                   command=lambda _: self.redraw_graph())
        style_menu.pack(fill=tk.X, padx=5, pady=2)

        # Arrow Size (only for directed graphs)
        self.arrow_size_frame = ttk.Frame(edge_style_frame)
        ttk.Label(self.arrow_size_frame, text="Arrow Size:").pack(anchor=tk.W, padx=5)
        ttk.Scale(self.arrow_size_frame, from_=5, to=30, variable=self.arrow_size,
                 command=lambda _: self.redraw_graph()).pack(fill=tk.X, padx=5, pady=2)
        self.arrow_size_frame.pack_forget()
        
        self.graph_type.trace_add('write', lambda *_: self.toggle_directed_options())
        
        # Community Detection Section
        comm_frame = ttk.LabelFrame(self.scrollable_frame, text="Community Detection", style='Header.TLabel')
        comm_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Community algorithm selection
        ttk.Label(comm_frame, text="Algorithm:").pack(anchor=tk.W, padx=5)
        comm_menu = ttk.OptionMenu(comm_frame, self.selected_community, "louvain", 
                                  "louvain", "girvan_newman")
        comm_menu.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(comm_frame, text="Run Community Detection", style='Success.TButton',
                  command=self.run_community_detection).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(comm_frame, text="Compare Algorithms", style='Primary.TButton',
                  command=self.compare_community_algorithms).pack(fill=tk.X, padx=5, pady=2)
        
        # Centrality Filters
        filter_frame = ttk.LabelFrame(self.scrollable_frame, text="Centrality Filters", style='Header.TLabel')
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.filter_threshold = tk.DoubleVar(value=0.0)
        ttk.Entry(filter_frame, textvariable=self.filter_threshold).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(filter_frame, text="Filter by Degree", style='Secondary.TButton',
                  command=lambda: self.filter_centrality('degree')).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(filter_frame, text="Filter by Betweenness", style='Secondary.TButton',
                  command=lambda: self.filter_centrality('betweenness')).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(filter_frame, text="Filter by Closeness", style='Secondary.TButton',
                  command=lambda: self.filter_centrality('closeness')).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(filter_frame, text="Run PageRank", style='Primary.TButton',
                  command=lambda: self.run_pagerank()).pack(fill=tk.X, padx=5, pady=2)
        
        # Metrics Section
        metrics_frame = ttk.LabelFrame(self.scrollable_frame, text="Metrics", style='Header.TLabel')
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(metrics_frame, text="Show Degree Distribution", style='Primary.TButton',
                  command=self.plot_degree_distribution).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(metrics_frame, text="Show Conductance", style='Primary.TButton',
                  command=self.show_conductance).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(metrics_frame, text="Show Modularity", style='Primary.TButton',
                  command=self.show_modularity).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(metrics_frame, text="Show NMI", style='Primary.TButton',
                  command=self.show_nmi).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(metrics_frame, text="Show ARI", style='Primary.TButton',
                  command=self.show_ari).pack(fill=tk.X, padx=5, pady=2)
        
        # Clear Button
        ttk.Button(self.scrollable_frame, text="Clear All", style='Danger.TButton',
                  command=self.clear_all).pack(fill=tk.X, padx=5, pady=10)

    def create_output_widgets(self):
        """Create widgets in the output frame"""
        self.output_text = tk.Text(self.output_frame, wrap=tk.WORD, font=('Consolas', 10))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(self.output_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.output_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.output_text.yview)

    def create_visualization_widgets(self):
        """Create widgets in the visualization frame"""
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.visualization_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def center_window(self):
        """Center the window on the screen"""
        self.master.update_idletasks()
        width = self.master.winfo_width()
        height = self.master.winfo_height()
        x = (self.master.winfo_screenwidth() // 2) - (width // 2)
        y = (self.master.winfo_screenheight() // 2) - (height // 2)
        self.master.geometry(f'{width}x{height}+{x}+{y}')

    def toggle_directed_options(self):
        """Toggle visibility of directed graph options"""
        if self.graph_type.get() == 'Directed Graph':
            self.arrow_size_frame.pack(fill=tk.X, padx=5, pady=2)
        else:
            self.arrow_size_frame.pack_forget()
        self.redraw_graph()

    def pick_node_color(self):
        """Open color picker for node color"""
        color = colorchooser.askcolor(title="Choose Node Color")[1]
        if color:
            self.node_single_color.set(color)
            self.node_color_preview.delete("all")
            self.node_color_preview.create_rectangle(0, 0, 20, 20, fill=color, outline="")
            self.redraw_graph()

    def pick_edge_color(self):
        """Open color picker for edge color"""
        color = colorchooser.askcolor(title="Choose Edge Color")[1]
        if color:
            self.edge_single_color.set(color)
            self.redraw_graph()

    def load_edge_file(self):
        """Load edge data from CSV file"""
        try:
            filepath = filedialog.askopenfilename(title="Select Edge CSV File", 
                                                filetypes=[("CSV files", "*.csv")])
            if filepath:
                self.edge_df = pd.read_csv(filepath)
                self.log_message(f"Loaded edge file: {filepath}")
                self.create_graph()
        except Exception as e:
            self.show_error("Error loading edge file", str(e))

    def load_node_file(self):
        """Load node data from CSV file"""
        try:
            filepath = filedialog.askopenfilename(title="Select Node CSV File", 
                                                filetypes=[("CSV files", "*.csv")])
            if filepath:
                self.node_df = pd.read_csv(filepath)
                self.log_message(f"Loaded node file: {filepath}")
                self.create_graph()
        except Exception as e:
            self.show_error("Error loading node file", str(e))

    def create_graph(self):
        """Create a NetworkX graph from the loaded data"""
        try:
            if self.edge_df is None:
                return False
            
            if self.graph_type.get() == 'Directed Graph':
                self.G = nx.from_pandas_edgelist(self.edge_df, 
                                               source="Source", 
                                               target="Target", 
                                               create_using=nx.DiGraph())
            else:
                self.G = nx.from_pandas_edgelist(self.edge_df, 
                                               source="Source", 
                                               target="Target", 
                                               create_using=nx.Graph())
            
            # Add node attributes if available
            if self.node_df is not None:
                for _, row in self.node_df.iterrows():
                    if row['ID'] in self.G.nodes():
                        for col in self.node_df.columns:
                            if col != 'ID':
                                self.G.nodes[row['ID']][col] = row[col]
            
            self.log_message(f"Created {self.graph_type.get()} with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
            self.redraw_graph()
            return True
        except Exception as e:
            self.show_error("Error creating graph", str(e))
            return False

    def apply_layout(self, G):
        """Apply the selected layout algorithm to the graph"""
        layout = self.selected_layout.get()
        seed = self.random_seed.get()
        
        if layout == "spring":
            return nx.spring_layout(G, seed=seed)
        elif layout == "circular":
            return nx.circular_layout(G)
        elif layout == "random":
            return nx.random_layout(G, seed=seed)
        elif layout == "shell":
            return nx.shell_layout(G)
        elif layout == "spectral":
            return nx.spectral_layout(G)
        elif layout == "tree":
            return self.tree_layout(G)
        elif layout == "radial":
            return self.radial_layout(G)
        else:
            return nx.spring_layout(G, seed=seed)

    def tree_layout(self, G):
        """Custom tree layout for hierarchical visualization"""
        if len(G) == 0:
            return {}
        
        # Choose a root (node with highest degree)
        root = max(G.nodes(), key=lambda x: G.degree(x))
        
        # Compute levels using BFS
        levels = {root: 0}
        queue = [root]
        while queue:
            current = queue.pop(0)
            for neighbor in G.neighbors(current):
                if neighbor not in levels:
                    levels[neighbor] = levels[current] + 1
                    queue.append(neighbor)
        
        # Organize nodes by level
        max_level = max(levels.values())
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        # Assign positions
        pos = {}
        for level in level_groups:
            nodes = level_groups[level]
            num_nodes = len(nodes)
            y = max_level - level  # Top to bottom
            x_positions = np.linspace(-num_nodes/2, num_nodes/2, num_nodes)
            for i, node in enumerate(nodes):
                pos[node] = (x_positions[i], y)
        
        return pos

    def radial_layout(self, G):
        """Custom radial layout for hierarchical visualization"""
        if len(G) == 0:
            return {}
        
        # Choose a root (node with highest degree)
        root = max(G.nodes(), key=lambda x: G.degree(x))
        
        # Compute levels from root
        levels = nx.single_source_shortest_path_length(G, root)
        max_level = max(levels.values()) if levels else 1
        
        # Organize nodes by level
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        pos = {}
        for level, nodes in level_groups.items():
            num_nodes = len(nodes)
            radius = level * 1.0  # Increase radius with level
            if num_nodes == 0:
                continue
            angles = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
            for i, node in enumerate(nodes):
                theta = angles[i]
                x = radius * np.cos(theta)
                y = radius * np.sin(theta)
                pos[node] = (x, y)
        
        return pos

    def redraw_graph(self):
        """Redraw the graph with current styling settings"""
        try:
            if self.G is None:
                return

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            pos = self.apply_layout(self.G)

            # Node styling
            node_size = [self.node_size.get() * (1 + self.G.degree(node)) / 10 for node in self.G.nodes()]
            marker_map = {'circle': 'o', 'square': 's', 'triangle': '^', 'diamond': 'D'}
            marker = marker_map.get(self.node_shape.get(), 'o')

            nx.draw_networkx_nodes(
                self.G, pos,
                node_color=self.node_single_color.get(),
                node_size=node_size,
                node_shape=marker,
                ax=ax
            )

            # Edge styling
            edge_style = self.edge_style.get()
            edge_width = self.edge_width.get()
            
            if self.graph_type.get() == 'Directed Graph':
                nx.draw_networkx_edges(
                    self.G, pos,
                    edge_color=self.edge_single_color.get(),
                    width=edge_width,
                    style=edge_style,
                    arrows=True,
                    arrowsize=self.arrow_size.get(),
                    ax=ax
                )
            else:
                nx.draw_networkx_edges(
                    self.G, pos,
                    edge_color=self.edge_single_color.get(),
                    width=edge_width,
                    style=edge_style,
                    ax=ax
                )

            # Node labels
            if self.node_label_visible.get() and len(self.G.nodes()) <= 100:
                nx.draw_networkx_labels(
                    self.G, pos,
                    font_size=self.node_label_size.get(),
                    ax=ax
                )

            plt.title("Network Visualization", pad=20)
            plt.axis('off')
            self.canvas.draw()

        except Exception as e:
            self.show_error("Error redrawing graph", str(e))

    def run_community_detection(self):
        """Run community detection and update visualization"""
        try:
            if not self.create_graph():
                return

            algorithm = self.selected_community.get()
            seed = self.random_seed.get()
            random.seed(seed)

            if algorithm == "louvain":
                partition = community_louvain.best_partition(self.G.to_undirected(), random_state=seed)
                title = 'Louvain Community Detection'
            elif algorithm == "girvan_newman":
                communities = next(girvan_newman(self.G.to_undirected()))
                partition = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        partition[node] = i
                title = 'Girvan-Newman Community Detection'

            self.show_community_stats(partition, title)
            self.log_message(f"{algorithm} algorithm completed successfully")

        except Exception as e:
            self.show_error("Error in community detection", str(e))

    def show_community_stats(self, partition, title):
        """Display community statistics in the output panel"""
        communities = {}
        for node, community in partition.items():
            if community not in communities:
                communities[community] = []
            communities[community].append(node)
        
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, f"{title} Results\n\n")
        self.output_text.insert(tk.END, f"Number of communities: {len(communities)}\n\n")
        
        for comm, nodes in communities.items():
            self.output_text.insert(tk.END, f"Community {comm}: {len(nodes)} nodes\n")
        
        # Calculate modularity
        mod = community_louvain.modularity(partition, self.G)
        self.output_text.insert(tk.END, f"\nModularity: {mod:.4f}\n")

    def compare_community_algorithms(self):
        """Compare different community detection algorithms"""
        try:
            if not self.create_graph():
                return
                
            seed = self.random_seed.get()
            random.seed(seed)
            
            # Run Louvain
            louvain_partition = community_louvain.best_partition(self.G.to_undirected(), random_state=seed)
            louvain_mod = community_louvain.modularity(louvain_partition, self.G)
            
            # Run Girvan-Newman (first level)
            gn_communities = next(girvan_newman(self.G.to_undirected()))
            gn_partition = {}
            for i, comm in enumerate(gn_communities):
                for node in comm:
                    gn_partition[node] = i
            gn_mod = community_louvain.modularity(gn_partition, self.G)
            
            # Calculate NMI if ground truth available
            nmi = None
            if self.node_df is not None and 'Class' in self.node_df.columns:
                ground_truth = dict(zip(self.node_df['ID'], self.node_df['Class']))
                louvain_truth = [louvain_partition[node] for node in self.G.nodes()]
                gn_truth = [gn_partition.get(node, -1) for node in self.G.nodes()]
                truth = [ground_truth[node] for node in self.G.nodes()]
                nmi_louvain = normalized_mutual_info_score(truth, louvain_truth)
                nmi_gn = normalized_mutual_info_score(truth, gn_truth)
                nmi = (nmi_louvain, nmi_gn)
            
            # Display results
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Community Detection Algorithm Comparison\n\n")
            self.output_text.insert(tk.END, f"Louvain Algorithm:\n")
            self.output_text.insert(tk.END, f"- Number of communities: {len(set(louvain_partition.values()))}\n")
            self.output_text.insert(tk.END, f"- Modularity: {louvain_mod:.4f}\n")
            if nmi:
                self.output_text.insert(tk.END, f"- NMI with ground truth: {nmi[0]:.4f}\n")
            
            self.output_text.insert(tk.END, f"\nGirvan-Newman Algorithm:\n")
            self.output_text.insert(tk.END, f"- Number of communities: {len(gn_communities)}\n")
            self.output_text.insert(tk.END, f"- Modularity: {gn_mod:.4f}\n")
            if nmi:
                self.output_text.insert(tk.END, f"- NMI with ground truth: {nmi[1]:.4f}\n")
            
            self.log_message("Community detection algorithms compared")
            
        except Exception as e:
            self.show_error("Error comparing algorithms", str(e))

    def plot_degree_distribution(self):
        """Plot the degree distribution of the graph"""
        try:
            if not self.create_graph():
                return
                
            degrees = [d for n, d in self.G.degree()]
            
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            ax.hist(degrees, bins=20, color='skyblue', edgecolor='black')
            ax.set_title('Degree Distribution')
            ax.set_xlabel('Degree')
            ax.set_ylabel('Frequency')
            ax.grid(True)
            
            self.canvas.draw()
            self.log_message("Degree distribution plotted")
            
        except Exception as e:
            self.show_error("Error plotting degree distribution", str(e))

    def filter_centrality(self, centrality_type):
        """Filter nodes based on centrality measure"""
        try:
            if not self.create_graph():
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
            else:
                raise ValueError("Invalid centrality type")
            
            filtered_nodes = [n for n in self.G.nodes() if centrality[n] >= threshold]
            filtered_G = self.G.subgraph(filtered_nodes)
            
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            pos = self.apply_layout(filtered_G)
            node_sizes = [self.node_size.get() * (1 + self.G.degree(n)) / 10 for n in filtered_nodes]
            
            nx.draw_networkx_nodes(filtered_G, pos, node_color=color, 
                                 node_size=node_sizes, ax=ax)
            
            if self.graph_type.get() == 'Directed Graph':
                nx.draw_networkx_edges(filtered_G, pos, arrows=True, ax=ax)
            else:
                nx.draw_networkx_edges(filtered_G, pos, ax=ax)
            
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

    def run_pagerank(self):
        """Run PageRank algorithm and display results"""
        try:
            if not self.create_graph():
                return
                
            pr = nx.pagerank(self.G)
            
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            pos = self.apply_layout(self.G)
            node_sizes = [self.node_size.get() * (1 + pr[node] * 100) for node in self.G.nodes()]
            
            nx.draw_networkx_nodes(self.G, pos, node_color='#6f42c1', 
                                 node_size=node_sizes, ax=ax)
            
            if self.graph_type.get() == 'Directed Graph':
                nx.draw_networkx_edges(self.G, pos, arrows=True, ax=ax)
            else:
                nx.draw_networkx_edges(self.G, pos, ax=ax)
            
            if len(self.G.nodes()) <= 50:
                nx.draw_networkx_labels(self.G, pos, font_size=8, ax=ax)
            
            plt.title('PageRank Results', pad=20)
            plt.axis('off')
            
            self.canvas.draw()
            
            # Display PageRank values in output
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "PageRank Results (Top 20 Nodes)\n\n")
            for node in sorted(pr.keys(), key=lambda x: pr[x], reverse=True)[:20]:
                self.output_text.insert(tk.END, f"Node {node}: {pr[node]:.4f}\n")
            
            self.log_message("PageRank algorithm completed")
            
        except Exception as e:
            self.show_error("Error running PageRank", str(e))

    def show_conductance(self):
        """Calculate and display conductance values"""
        try:
            if not self.create_graph():
                return
                
            partition = community_louvain.best_partition(self.G.to_undirected())
            conductance_values = self.calculate_conductance(self.G, partition)
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Community Conductance Values:\n\n")
            
            for community, conductance in conductance_values.items():
                self.output_text.insert(tk.END, f"{community}: {conductance:.4f}\n")
            
            avg_conductance = sum(conductance_values.values()) / len(conductance_values)
            self.output_text.insert(tk.END, f"\nAverage Conductance: {avg_conductance:.4f}")
            
            self.log_message("Conductance values calculated")
            
        except Exception as e:
            self.show_error("Error calculating conductance", str(e))

    def calculate_conductance(self, G, partition):
        """Calculate conductance for each community"""
        communities = {}
        for node, community in partition.items():
            if community not in communities:
                communities[community] = []
            communities[community].append(node)
        
        conductance_values = {}
        for community, nodes in communities.items():
            Eoc = 0  # Edges from community to outside
            Ec = 0   # Edges within community
            
            for node in nodes:
                for neighbor in G.neighbors(node):
                    if neighbor in nodes:
                        Ec += 1
                    else:
                        Eoc += 1
            
            # Each edge is counted twice (once for each node)
            Ec = Ec // 2
            Eoc = Eoc // 2
            
            if Ec == 0:
                conductance = 1.0
            else:
                conductance = Eoc / (2 * Ec + Eoc)
            
            conductance_values[f"Community {community}"] = conductance
        
        return conductance_values

    def show_modularity(self):
        """Calculate and display modularity"""
        try:
            if not self.create_graph():
                return
                
            partition = community_louvain.best_partition(self.G.to_undirected())
            communities = {}
            for node, community in partition.items():
                if community not in communities:
                    communities[community] = []
                communities[community].append(node)
            
            mod = community_louvain.modularity(self.G, communities.values())
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Modularity: {mod:.4f}\n\n")
            self.output_text.insert(tk.END, f"Number of communities: {len(communities)}")
            
            self.log_message(f"Modularity calculated: {mod:.4f}")
            
        except Exception as e:
            self.show_error("Error calculating modularity", str(e))

    def show_nmi(self):
        """Calculate and display Normalized Mutual Information"""
        try:
            if not self.create_graph() or self.node_df is None or 'Class' not in self.node_df.columns:
                raise ValueError("Node data with ground truth classes is required")
                
            partition = community_louvain.best_partition(self.G.to_undirected())
            ground_truth = dict(zip(self.node_df['ID'], self.node_df['Class']))
            
            # Align the partitions
            detected = [partition[node] for node in self.G.nodes()]
            truth = [ground_truth[node] for node in self.G.nodes()]
            
            nmi = normalized_mutual_info_score(truth, detected)
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Normalized Mutual Information (NMI): {nmi:.4f}")
            
            self.log_message(f"NMI calculated: {nmi:.4f}")
            
        except Exception as e:
            self.show_error("Error calculating NMI", str(e))

    def show_ari(self):
        """Calculate and display Adjusted Rand Index"""
        try:
            if not self.create_graph() or self.node_df is None or 'Class' not in self.node_df.columns:
                raise ValueError("Node data with ground truth classes is required")
                
            partition = community_louvain.best_partition(self.G.to_undirected())
            ground_truth = dict(zip(self.node_df['ID'], self.node_df['Class']))
            
            # Align the partitions
            detected = [partition[node] for node in self.G.nodes()]
            truth = [ground_truth[node] for node in self.G.nodes()]
            
            ari = adjusted_rand_score(truth, detected)
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"Adjusted Rand Index (ARI): {ari:.4f}")
            
            self.log_message(f"ARI calculated: {ari:.4f}")
            
        except Exception as e:
            self.show_error("Error calculating ARI", str(e))

    def clear_all(self):
        """Clear all data and visualizations"""
        self.figure.clear()
        self.canvas.draw()
        self.output_text.delete(1.0, tk.END)
        self.edge_df = None
        self.node_df = None
        self.G = None
        self.log_message("All data cleared")

    def log_message(self, message):
        """Add a message to the output log"""
        self.output_text.insert(tk.END, f"[LOG] {message}\n")
        self.output_text.see(tk.END)

    def show_error(self, title, message):
        """Show an error message"""
        error_msg = f"{title}:\n{message}"
        self.log_message(error_msg)
        messagebox.showerror(title, message)

def main():
    root = tk.Tk()
    root.geometry("1200x800")
    app = NetworkAnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"Unhandled exception: {str(e)}\n{traceback.format_exc()}"
        messagebox.showerror("Critical Error", error_msg)
        sys.exit(1) 