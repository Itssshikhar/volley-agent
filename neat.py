import jax
import jax.numpy as jnp
from jax import random
import functools
import gymnasium as gym
import slimevolleygym

# Simplified Slime Volley environment
# class SlimeVolley:
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         # Simplified reset: just return a random observation
#         return random.normal(random.PRNGKey(0), (8,))

#     def step(self, action):
#         # Simplified step: return random observation, reward, and done
#         key = random.PRNGKey(0)
#         return random.normal(key, (8,)), random.uniform(key, ()), random.uniform(key, ()) < 0.1

#     def get_observation(self):
#         # Return a random observation
#         return random.normal(random.PRNGKey(0), (8,))

# NEAT implementation
class Genome:
    def __init__(self, input_size, output_size, hidden_nodes, connections):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_nodes = jnp.array(hidden_nodes, dtype=jnp.int32)
        self.connections = jnp.array(connections, dtype=jnp.float32)

def initialize_genome(key, input_size, output_size):
    key1, key2 = random.split(key)
    hidden_nodes = jnp.array([], dtype=jnp.int32)
    connections = jnp.zeros((input_size * output_size, 4), dtype=jnp.float32)
    connections = connections.at[:, :2].set(jnp.mgrid[:input_size, :output_size].reshape(2, -1).T)
    connections = connections.at[:, 2].set(random.normal(key2, (input_size * output_size,)))
    connections = connections.at[:, 3].set(1)
    return Genome(input_size, output_size, hidden_nodes, connections)

def initialize_population(key, pop_size, input_size, output_size):
    keys = random.split(key, pop_size)
    return [initialize_genome(k, input_size, output_size) for k in keys]

@jax.jit
def add_connection(key, genome):
    new_connection = random.uniform(key, (1, 4))
    return Genome(genome.input_size, genome.output_size, genome.hidden_nodes,
                  jnp.concatenate([genome.connections, new_connection]))

@jax.jit
def add_node(key, genome):
    key1, key2 = random.split(key)
    conn_to_split = random.choice(key1, len(genome.connections))
    new_node = jnp.max(genome.hidden_nodes) + 1 if len(genome.hidden_nodes) > 0 else 0
    new_connections = jnp.array([
        [genome.connections[conn_to_split, 0], new_node, 1.0, 1],
        [new_node, genome.connections[conn_to_split, 1], genome.connections[conn_to_split, 2], 1]
    ])
    return Genome(genome.input_size, genome.output_size,
                  jnp.append(genome.hidden_nodes, new_node),
                  jnp.concatenate([genome.connections, new_connections]))

def crossover(key, genome1, genome2, fitness_func):
    key1, key2 = random.split(key)
    fitness1 = fitness_func(genome1)
    fitness2 = fitness_func(genome2)
    fitter_parent = jax.lax.cond(fitness1 > fitness2, lambda: genome1, lambda: genome2)
    less_fit_parent = jax.lax.cond(fitness1 > fitness2, lambda: genome2, lambda: genome1)
    
    mask = random.uniform(key1, (len(fitter_parent.connections),)) < 0.7
    child_connections = jnp.where(mask[:, None], fitter_parent.connections, less_fit_parent.connections)
    
    return Genome(fitter_parent.input_size, fitter_parent.output_size,
                  jnp.unique(jnp.concatenate([fitter_parent.hidden_nodes, less_fit_parent.hidden_nodes])),
                  child_connections)

@functools.partial(jax.jit, static_argnums=(0, 1))
def compute_output(input_size, output_size, hidden_nodes, connections, input_vector):
    hidden_size = jnp.max(hidden_nodes) + 1 if hidden_nodes.size > 0 else 0
    hidden = jnp.zeros(max(hidden_size, 1))
    output = jnp.zeros(output_size)
    
    def body_fun(i, val):
        hidden, output = val
        conn = connections[i]
        
        # Source selection
        is_input = conn[0] < input_size
        is_hidden = (conn[0] >= input_size) & (conn[0] < input_size + hidden_size)
        
        input_source = jnp.where(is_input, input_vector[jnp.array(conn[0], dtype=jnp.int32)], 0.0)
        hidden_source = jnp.where(is_hidden & (hidden_size > 0), hidden[jnp.array(conn[0] - input_size, dtype=jnp.int32)], 0.0)
        source = jnp.where(is_input, input_source, hidden_source)
        
        # Hidden layer update
        is_to_hidden = (conn[1] >= input_size) & (conn[1] < input_size + hidden_size)
        hidden_index = jnp.array(conn[1] - input_size, dtype=jnp.int32)
        hidden = jnp.where(is_to_hidden & (hidden_size > 0), hidden.at[hidden_index].add(source * conn[2]), hidden)
        
        # Output layer update
        is_to_output = conn[1] >= input_size + hidden_size
        output_index = jnp.array(conn[1] - (input_size + hidden_size), dtype=jnp.int32)
        output = jnp.where(is_to_output, output.at[output_index].add(source * conn[2]), output)
        
        return hidden, output
    
    _, output = jax.lax.fori_loop(0, connections.shape[0], body_fun, (hidden, output))
    
    return jax.nn.sigmoid(output)

def fitness_function(genome, env, num_episodes=3):
    total_reward = 0
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = jnp.array(compute_output(int(genome.input_size), int(genome.output_size), genome.hidden_nodes, genome.connections, jnp.array(obs)))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    return total_reward / num_episodes 

def jitted_fitness_function(genome, env, num_episodes):
    return fitness_function(genome, env, num_episodes)

def evolution_step(key, population, env, num_episodes, population_size):
    keys = random.split(key, 4)
    
    fitnesses = [fitness_function(genome, env, num_episodes) for genome in population]
    
    parent_indices = random.choice(keys[0], population_size, shape=(population_size, 2), p=jax.nn.softmax(jnp.array(fitnesses)))
    parents1 = [population[i] for i in parent_indices[:, 0]]
    parents2 = [population[i] for i in parent_indices[:, 1]]
    
    offspring = [crossover(k, p1, p2, lambda g: fitness_function(g, env, num_episodes))
                 for k, p1, p2 in zip(random.split(keys[1], population_size), parents1, parents2)]
    
    mutation_mask = random.uniform(keys[2], (population_size,)) < 0.3
    mutated_offspring = [
        add_connection(k, g) if random.uniform(k) < 0.5 else add_node(k, g)
        for k, g, m in zip(random.split(keys[3], population_size), offspring, mutation_mask) if m
    ]
    
    return [o if not m else m_o for o, m, m_o in zip(offspring, mutation_mask, mutated_offspring + offspring)] 

# Training process
def train_neat_agent():
    env = gym.make('SlimeVolley-v0')

    key = random.PRNGKey(0)
    population_size = 100
    input_size = 8  # Assuming 8 inputs from the environment
    output_size = 4  # 4 possible actions
    num_generations = 100
    num_episodes = 3

    population = initialize_population(key, population_size, input_size, output_size)

    for generation in range(num_generations):
        key, subkey = random.split(key)
        population = evolution_step(subkey, population, env, num_episodes, population_size)

        fitnesses = jax.vmap(jitted_fitness_function, in_axes=(0, None, None))(population, env, num_episodes)
        best_fitness = jnp.max(fitnesses)
        best_agent = population[jnp.argmax(fitnesses)]

        print(f"Generation {generation}, Best fitness: {best_fitness}")

    final_fitness = fitness_function(best_agent, env, num_episodes=10)
    print(f"Final best agent fitness: {final_fitness}")

    return best_agent

# Run the training
best_agent = train_neat_agent()

# Test the agent run 
def test_agent(agent, env, num_episodes=5):
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            action = jnp.array(compute_output(int(agent.input_size), int(agent.output_size), agent.hidden_nodes, agent.connections, jnp.array(obs)))
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

# Create a rendering environment and test the best agent
render_env = gym.make("SlimeVolley-v0", render_mode="human")
test_agent(best_agent, render_env)
render_env.close()
