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
        
        # Initialize UI
        self.setup_ui()
        
    def setup_ui(self):
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', font=('Arial', 10), padding=5)
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'))
        
        # Main frames
        self.control_frame = ttk.Frame(self.root, width=300, padding=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Visualization canvas
        self.figure = plt.figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Build UI sections
        self.setup_data_loading()
        self.setup_visualization_controls()
        self.setup_analysis_controls()
        self.setup_filter_controls()
        
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
                    values=["spring", "circular", "kamada-kawai", "spectral", "shell"]).pack(fill=tk.X)
        
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
        
        ttk.Button(frame, text="Update Visualization", command=self.update_visualization).pack(fill=tk.X, pady=5)
        ttk.Button(frame, text="Reapply Layout", command=self.force_new_layout).pack(fill=tk.X)
        
    def setup_analysis_controls(self):
        frame = ttk.LabelFrame(self.control_frame, text="Analysis Tools", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(frame, text="Run Louvain Community Detection", 
                  command=lambda: self.run_community_detection("louvain")).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Run Girvan-Newman Community Detection", 
                  command=lambda: self.run_community_detection("girvan")).pack(fill=tk.X, pady=2)
        
        ttk.Button(frame, text="Calculate PageRank", command=self.calculate_pagerank).pack(fill=tk.X, pady=2)
        ttk.Button(frame, text="Calculate Betweenness Centrality", command=self.calculate_betweenness).pack(fill=tk.X, pady=2)
        
    def setup_filter_controls(self):
        frame = ttk.LabelFrame(self.control_frame, text="Filter Nodes", padding=10)
        frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(frame, text="Filter By:").pack(anchor=tk.W)
        self.filter_var = tk.StringVar()
        ttk.Combobox(frame, textvariable=self.filter_var,
                    values=["Degree Centrality", "Betweenness Centrality", "Community"]).pack(fill=tk.X)
        
        ttk.Label(frame, text="Threshold/Community ID:").pack(anchor=tk.W)
        self.threshold_entry = ttk.Entry(frame)
        self.threshold_entry.pack(fill=tk.X)
        
        ttk.Button(frame, text="Apply Filter", command=self.apply_filter).pack(fill=tk.X, pady=5)
        ttk.Button(frame, text="Reset Graph", command=self.reset_graph).pack(fill=tk.X)
        
    def choose_node_color(self):
        color = colorchooser.askcolor(title="Choose Node Color", initialcolor=self.custom_color)
        if color[1]:  # User didn't cancel
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
        
        nx.draw_networkx_edges(self.G, self.layout_pos, ax=ax, width=self.edge_width)
        
        if len(self.G.nodes()) <= 50:
            nx.draw_networkx_labels(self.G, self.layout_pos, ax=ax, font_size=8)
        
        if self.current_layout in ["shell", "circular"]:
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
        
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
                pos = nx.circular_layout(self.G)
                return {k: np.array(v)*0.8 for k, v in pos.items()}
            elif layout == "kamada-kawai":
                return nx.kamada_kawai_layout(self.G)
            elif layout == "spectral":
                return nx.spectral_layout(self.G)
            elif layout == "shell":
                shells = [[] for _ in range(min(4, len(self.G.nodes())))]
                for i, node in enumerate(self.G.nodes()):
                    shells[i % len(shells)].append(node)
                pos = nx.shell_layout(self.G, nlist=shells)
                return {k: np.array(v)*0.8 for k, v in pos.items()}
            else:
                return nx.spring_layout(self.G, seed=self.layout_seed)
        except Exception as e:
            messagebox.showerror("Layout Error", str(e))
            return nx.spring_layout(self.G, seed=self.layout_seed)
    
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
            
            # Force community coloring and refresh
            self.color_var.set("community")
            self.show_community_metrics()
            self.update_visualization()
            messagebox.showinfo("Success", message)
            
        except Exception as e:
            messagebox.showerror("Error", f"Community detection failed: {str(e)}")
    
    def show_community_metrics(self):
        if not self.communities:
            return
            
        metrics = []
        communities = defaultdict(list)
        for node, comm_id in self.communities.items():
            communities[comm_id].append(node)
        
        metrics.append(f"Number of communities: {len(communities)}")
        metrics.append(f"Modularity: {nx.community.modularity(self.G, communities.values()):.3f}")
        
        if hasattr(self, 'node_attributes') and 'class' in next(iter(self.node_attributes.values()), {}):
            ground_truth = {node: data.get('class', 0) for node, data in self.G.nodes(data=True)}
            nmi = normalized_mutual_info_score(
                list(self.communities.values()),
                list(ground_truth.values())
            )
            metrics.append(f"Normalized Mutual Information: {nmi:.3f}")
        
        conductances = []
        for comm in communities.values():
            cut = nx.cut_size(self.G, comm)
            volume = nx.volume(self.G, comm)
            conductances.append(cut / volume if volume != 0 else 0)
        metrics.append(f"Average Conductance: {np.mean(conductances):.3f}")
        
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
    
    def apply_filter(self):
        if self.G is None:
            messagebox.showerror("Error", "No graph loaded")
            return
            
        filter_type = self.filter_var.get()
        threshold = self.threshold_entry.get()
        
        if not filter_type:
            messagebox.showerror("Error", "Select a filter type first")
            return
            
        try:
            if filter_type == "Community":
                if not threshold.isdigit():
                    messagebox.showerror("Error", "Enter a valid community ID")
                    return
                    
                comm_id = int(threshold)
                filtered_nodes = [node for node, cid in self.communities.items() if cid == comm_id]
                self.G = self.G.subgraph(filtered_nodes)
                
            else:
                if not threshold.replace('.', '').isdigit():
                    messagebox.showerror("Error", "Enter a valid threshold")
                    return
                    
                threshold = float(threshold)
                
                if filter_type == "Degree Centrality":
                    scores = nx.degree_centrality(self.G)
                elif filter_type == "Betweenness Centrality":
                    scores = nx.betweenness_centrality(self.G)
                
                filtered_nodes = [node for node, score in scores.items() if score >= threshold]
                self.G = self.G.subgraph(filtered_nodes)
            
            self.layout_pos = None
            self.update_visualization()
            messagebox.showinfo("Success", f"Filter applied: {len(self.G.nodes())} nodes remaining")
            
        except Exception as e:
            messagebox.showerror("Error", f"Filtering failed: {str(e)}")
    
    def reset_graph(self):
        try:
            if self.original_graph is not None:
                self.G = self.original_graph.copy()
                self.layout_pos = None
                self.update_visualization()
                messagebox.showinfo("Success", "Graph reset to original state")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset graph: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SocialNetworkAnalyzer(root)
    root.mainloop()