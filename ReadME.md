# NEAT Slime Volley Implementation

A JAX-based implementation of NeuroEvolution of Augmenting Topologies (NEAT) algorithm for training agents to play Slime Volley.

## Overview

This implementation uses first principles to evolve neural networks that can play Slime Volley. The project leverages JAX for efficient computation and GPU acceleration.

![slime-agent](slime-agent.gif)

## Features

- First principles NEAT implementation
- JAX optimization with JIT compilation
- Genome representation with dynamic topology
- Speciation for maintaining population diversity
- Gymnasium (Gym) environment integration

## Requirements

- JAX
- Gymnasium
- NumPy
- SlimeVolley-v0 environment

## Implementation Details

### Core Components

1. **Genome Representation**: Flexible neural network structure using JAX arrays
2. **Population Management**: Handles initialization and evolution of genomes
3. **Network Modification**: Supports adding nodes and connections
4. **Crossover & Mutation**: Implements genetic operations for evolution
5. **Fitness Evaluation**: Assesses agent performance in the environment

### Key Optimizations

- JIT compilation for performance-critical functions
- Vectorized operations using JAX
- GPU acceleration support

## Usage

```python
# Initialize and train an agent
env = gym.make('SlimeVolley-v0')
agent = train_neat_agent()

# Test the trained agent
test_agent(agent, env, num_episodes=5)
```
