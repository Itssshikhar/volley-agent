import jax
import jax.numpy as jnp
from jax import grad, jit, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
import networkx as nx

def sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def relu(x):
    return jnp.maximum(0, x)

class NEATGenome:
    def __init__(self, input_size, output_size, rng_key):
        self.input_size = input_size
        self.output_size = output_size
        self.rng_key = rng_key
        self.connections = []
        self.nodes = []
        self.initialize_network()

    def initialize_network(self):
        self.connections = random.uniform(self.rng_key, (self.input_size * self.output_size,))
        self.nodes = random.uniform(self.rng_key, (self.input_size + self.output_size,))

    def forward(self, inputs):
        activations = jnp.array(inputs)
        idx = 0
        for i in range(self.input_size):
            for j in range(self.output_size):
                activations = relu(jnp.dot(activations, self.connections[idx]) + self.nodes[i])
                idx += 1
        return sigmoid(activations)

    def mutate(self):
        mutation_rate = 0.1
        self.connections += random.uniform(self.rng_key, self.connections.shape) * mutation_rate
        self.nodes += random.uniform(self.rng_key, self.nodes.shape) * mutation_rate

    def copy(self):
        new_genome = NEATGenome(self.input_size, self.output_size, self.rng_key)
        new_genome.connections = self.connections.copy()
        new_genome.nodes = self.nodes.copy()
        return new_genome

    def get_fitness(self, data, labels):
        preds = self.forward(data)
        loss = -jnp.mean(labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds))
        penalty = len(self.connections) * 0.03
        return -loss * (1 + penalty)

class NEATTrainer:
    def __init__(self, population_size, input_size, output_size, rng_key):
        self.population_size = population_size
        self.input_size = input_size
        self.output_size = output_size
        self.rng_key = rng_key
        self.population = [NEATGenome(input_size, output_size, rng_key) for _ in range(population_size)]

    def evolve(self, data, labels):
        self.population.sort(key=lambda g: g.get_fitness(data, labels), reverse=True)
        survivors = self.population[:self.population_size // 2]
        offspring = [parent.copy() for parent in survivors]
        for genome in offspring:
            genome.mutate()
        self.population = survivors + offspring

    def get_best_genome(self, data, labels):
        return max(self.population, key=lambda g: g.get_fitness(data, labels))
    
    def apply_fitness_func(self, fitness_func, data, labels):
        for genome in self.population:
            fitness_func(genome, data, labels)

class DataSet:
    def __init__(self, rng_key, n_size=200, n_test_size=200, noise_level=0.5, n_batch=10):
        self.rng_key = rng_key
        self.n_size = n_size
        self.n_test_size = n_test_size
        self.noise_level = noise_level
        self.n_batch = n_batch
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.data_batch = None
        self.label_batch = None

    def shuffle_data_list(self, data_list):
        return random.permutation(self.rng_key, jnp.array(data_list))

    def generate_xor_data(self, n_points=None, noise=None):
        n_points = n_points or self.n_size
        noise = noise or self.noise_level
        data_list = []

        for _ in range(n_points):
            x = random.uniform(self.rng_key, (), minval=-5.0, maxval=5.0) + random.normal(self.rng_key, ()) * noise
            y = random.uniform(self.rng_key, (), minval=-5.0, maxval=5.0) + random.normal(self.rng_key, ()) * noise
            label = 1 if (x > 0 and y > 0) or (x < 0 and y < 0) else 0
            data_list.append([x, y, label])

        return data_list

    def generate_spiral_data(self, n_points=None, noise=None):
        n_points = n_points or self.n_size
        noise = noise or self.noise_level
        data_list = []

        def gen_spiral(delta_t, label):
            for i in range(n_points // 2):
                r = i / (n_points // 2) * 6.0
                t = 1.75 * i / (n_points // 2) * 2 * jnp.pi + delta_t
                x = r * jnp.sin(t) + random.uniform(self.rng_key, (), minval=-1, maxval=1) * noise
                y = r * jnp.cos(t) + random.uniform(self.rng_key, (), minval=-1, maxval=1) * noise
                data_list.append([x, y, label])

        gen_spiral(0, 0)
        gen_spiral(jnp.pi, 1)
        return data_list

    def generate_gaussian_data(self, n_points=None, noise=None):
        n_points = n_points or self.n_size
        noise = noise or self.noise_level
        data_list = []

        def gen_gaussian(xc, yc, label):
            for _ in range(n_points // 2):
                x = random.normal(self.rng_key, ()) * noise + xc
                y = random.normal(self.rng_key, ()) * noise + yc
                data_list.append([x, y, label])

        gen_gaussian(2, 2, 1)
        gen_gaussian(-2, -2, 0)
        return data_list

    def generate_circle_data(self, n_points=None, noise=None):
        n_points = n_points or self.n_size
        noise = noise or self.noise_level
        data_list = []
        radius = 5.0

        def get_circle_label(x, y):
            return 1 if (x**2 + y**2) < (radius * 0.5)**2 else 0

        for _ in range(n_points // 2):
            r = random.uniform(self.rng_key, (), minval=0, maxval=radius * 0.5)
            angle = random.uniform(self.rng_key, (), minval=0, maxval=2 * jnp.pi)
            x = r * jnp.sin(angle)
            y = r * jnp.cos(angle)
            noise_x = random.uniform(self.rng_key, (), minval=-radius, maxval=radius) * noise / 3
            noise_y = random.uniform(self.rng_key, (), minval=-radius, maxval=radius) * noise / 3
            label = get_circle_label(x, y)
            data_list.append([x + noise_x, y + noise_y, label])

        for _ in range(n_points // 2):
            r = random.uniform(self.rng_key, (), minval=radius * 0.75, maxval=radius)
            angle = random.uniform(self.rng_key, (), minval=0, maxval=2 * jnp.pi)
            x = r * jnp.sin(angle)
            y = r * jnp.cos(angle)
            noise_x = random.uniform(self.rng_key, (), minval=-radius, maxval=radius) * noise / 3
            noise_y = random.uniform(self.rng_key, (), minval=-radius, maxval=radius) * noise / 3
            label = get_circle_label(x, y)
            data_list.append([x + noise_x, y + noise_y, label])

        return data_list

    def convert_data(self, data_list):
        data = jnp.array([[point[0], point[1]] for point in data_list])
        labels = jnp.array([[point[2]] for point in data_list])
        return data, labels

    def generate_random_data(self, choice):
        if choice == 0:
            train_list = self.generate_circle_data()
            test_list = self.generate_circle_data(self.n_test_size)
        elif choice == 1:
            train_list = self.generate_xor_data()
            test_list = self.generate_xor_data(self.n_test_size)
        elif choice == 2:
            train_list = self.generate_gaussian_data()
            test_list = self.generate_gaussian_data(self.n_test_size)
        elif choice == 3:
            train_list = self.generate_spiral_data()
            test_list = self.generate_spiral_data(self.n_test_size)
        else:
            raise ValueError("Invalid choice for dataset generation")

        self.train_data, self.train_labels = self.convert_data(self.shuffle_data_list(train_list))
        self.test_data, self.test_labels = self.convert_data(self.shuffle_data_list(test_list))
        self.data_batch, self.label_batch = self.generate_mini_batch()

    def generate_mini_batch(self):
        indices = random.choice(self.rng_key, self.n_size, shape=(self.n_batch,))
        data_batch = self.train_data[indices]
        label_batch = self.train_labels[indices]
        return data_batch, label_batch

# Example usage:
def fitness_func(genome, data, labels, backprop_mode=False, n_cycles=1):
    def loss_fn(connections):
        preds = genome.forward(data)
        loss = -jnp.mean(labels * jnp.log(preds) + (1 - labels) * jnp.log(1 - preds))
        return loss

    if backprop_mode:
        for _ in range(n_cycles):
            grads = grad(loss_fn)(genome.connections)
            genome.connections -= grads * 0.01

    return genome.get_fitness(data, labels)

class NetworkVisualizer:
    def __init__(self, trainer, dataset):
        self.trainer = trainer
        self.dataset = dataset
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.G = nx.DiGraph()
        self.pos = {}
        self.generation = 0

    def update_graph(self, genome):
        self.G.clear()
        input_nodes = range(genome.input_size)
        output_nodes = range(genome.input_size, genome.input_size + genome.output_size)
        hidden_nodes = range(genome.input_size + genome.output_size, len(genome.nodes))

        for i in input_nodes:
            self.G.add_node(i, layer=0)
        for i in output_nodes:
            self.G.add_node(i, layer=2)
        for i, node in enumerate(hidden_nodes):
            self.G.add_node(node, layer=1)

        idx = 0
        for i in range(genome.input_size):
            for j in range(genome.output_size):
                weight = genome.connections[idx]
                self.G.add_edge(i, genome.input_size + j, weight=weight)
                idx += 1

        self.pos = {}
        layers = nx.get_node_attributes(self.G, 'layer')
        layer_counts = {0: 0, 1: 0, 2: 0}
        for node, layer in layers.items():
            x = layer
            y = layer_counts[layer]
            self.pos[node] = (x, y)
            layer_counts[layer] += 1

        for layer in range(3):
            nodes_in_layer = [n for n, l in layers.items() if l == layer]
            count = len(nodes_in_layer)
            for i, node in enumerate(nodes_in_layer):
                self.pos[node] = (layer, (i - (count-1)/2) / max(count-1, 1))

    def draw_graph(self):
        self.ax.clear()
        edge_weights = [self.G[u][v]['weight'] for u, v in self.G.edges()]
        norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
        edge_colors = [plt.cm.bwr(norm(abs(weight))) for weight in edge_weights]
        edge_widths = [abs(weight) * 2 for weight in edge_weights]

        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, node_size=500, node_color='lightblue')
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edge_color=edge_colors, width=edge_widths, 
                               arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_labels(self.G, self.pos, ax=self.ax, font_size=8)

        self.ax.set_xlim(-0.5, 2.5)
        self.ax.set_ylim(-2, 2)
        self.ax.axis('off')
        self.ax.set_title(f'Generation: {self.generation}')

    def animate(self, frame):
        self.generation = frame
        best_genome = max(self.trainer.population, key=lambda g: g.get_fitness(self.dataset.train_data, self.dataset.train_labels))
        self.update_graph(best_genome)
        self.draw_graph()
        self.trainer.evolve(self.dataset.train_data, self.dataset.train_labels)
        return self.ax
    
############################## VISUALIZING THE GENOME ################################

def visualize_genome(genome, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    G = nx.DiGraph()
    
    input_nodes = range(genome.input_size)
    output_nodes = range(genome.input_size, genome.input_size + genome.output_size)
    hidden_nodes = range(genome.input_size + genome.output_size, len(genome.nodes))

    # Add nodes to the graph with their type as an attribute
    for i in input_nodes:
        G.add_node(i, type='input')
    for i in output_nodes:
        G.add_node(i, type='output')
    for i in hidden_nodes:
        G.add_node(i, type='hidden')

    # Add edges to the graph
    idx = 0
    for i in range(genome.input_size):
        for j in range(genome.output_size):
            weight = genome.connections[idx]
            G.add_edge(i, genome.input_size + j, weight=weight)
            idx += 1

    # Define positions for each type of node using shell layout
    pos = nx.shell_layout(G, nlist=[input_nodes, hidden_nodes, output_nodes])

    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
    edge_colors = [plt.cm.bwr(norm(weight)) for weight in edge_weights]
    
    # Draw the nodes and edges
    node_colors = [G.nodes[n]['type'] for n in G.nodes()]
    color_map = {'input': 'lightgreen', 'hidden': 'lightblue', 'output': 'lightcoral'}
    node_colors = [color_map[node_type] for node_type in node_colors]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700, node_color=node_colors)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, connectionstyle='arc3,rad=0.2')
    nx.draw_networkx_labels(G, pos, ax=ax)
    
    edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)
    
    ax.axis('off')

def visualize_population(trainer, dataset, generation):
    fig, axes = plt.subplots(1, len(trainer.population), figsize=(20, 5))
    fig.suptitle(f'Generation {generation}', fontsize=16)
    for ax, genome in zip(axes, trainer.population):
        visualize_genome(genome, ax=ax)
    plt.show()

def visualize_performance(trainer, dataset, fitness_func, generations=10, dataset_name=""):
    for generation in range(generations):
        print(f"Generation {generation} - Dataset: {dataset_name}")
        trainer.apply_fitness_func(fitness_func, dataset.train_data, dataset.train_labels)
        visualize_population(trainer, dataset, generation)
        trainer.evolve(dataset.train_data, dataset.train_labels)

############################## VISUALIZING THE GENOME ################################

def train_and_visualize(trainer, dataset, generations=10):
    visualizer = NetworkVisualizer(trainer, dataset)
    animation = FuncAnimation(visualizer.fig, visualizer.animate, frames=generations, interval=500, repeat=False)
    plt.show()

# Example usage:
rng_key = random.PRNGKey(0)
# dataset = DataSet(rng_key)
dataset = ["XOR", "Spiral", "Gaussians", "Circle"]
# ataset.generate_random_data(3)  # 0: circle, 1: XOR, 2: gaussians, 3: spiral

for i, name in enumerate(dataset):
    dataset = DataSet(rng_key)
    dataset.generate_random_data(i)
    neat_trainer = NEATTrainer(population_size=5, input_size=2, output_size=1, rng_key=rng_key)
    visualize_performance(neat_trainer, dataset, fitness_func, generations=10, dataset_name=name)

neat_trainer = NEATTrainer(population_size=20, input_size=2, output_size=1, rng_key=rng_key)

# Apply fitness function to the initial population
neat_trainer.apply_fitness_func(fitness_func, dataset.train_data, dataset.train_labels)

train_and_visualize(neat_trainer, dataset, generations=10)

# Evolve and backprop
for _ in range(10):
    neat_trainer.evolve(dataset.train_data, dataset.train_labels)
    best_genome = neat_trainer.get_best_genome(dataset.train_data, dataset.train_labels)
    fitness_func(best_genome, dataset.train_data, dataset.train_labels, backprop_mode=True, n_cycles=600)

print("Best genome fitness:", best_genome.get_fitness(dataset.train_data, dataset.train_labels))
print("Train Data", dataset.train_data)
print("Train Labels:", dataset.train_labels)