import numpy as np
import networkx as nx
from typing import List, Tuple, Set
import random


class Bot2Vec:
    def __init__(
            self,
            learning_rate: float = 0.025,
            embedding_dim: int = 128,
            max_iter: int = 10,
            walk_length: int = 80,
            num_walks: int = 10,
            window_size: int = 10,
            neg_batch_size: int = 5,
            p: float = 1.0,
            q: float = 1.0,
            r: float = 1.0
    ):
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.max_iter = max_iter
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.neg_batch_size = neg_batch_size
        self.p = p
        self.q = q
        self.r = r

    def _random_walk(self, G: nx.Graph, start_node: int) -> List[int]:
        """Generate a random walk starting from start_node."""
        walk = [start_node]

        for _ in range(self.walk_length - 1):
            cur = walk[-1]
            neighbors = list(G.neighbors(cur))

            if not neighbors:
                break

            # Biased random walk based on p, q, r parameters
            if len(walk) > 1:
                prev = walk[-2]
                probs = []
                for neighbor in neighbors:
                    if neighbor == prev:  # Return parameter p
                        probs.append(1 / self.p)
                    elif G.has_edge(prev, neighbor):  # Parameter q
                        probs.append(1 / self.q)
                    else:  # Parameter r
                        probs.append(1 / self.r)

                probs = np.array(probs)
                probs = probs / probs.sum()
                next_node = np.random.choice(neighbors, p=probs)
            else:
                next_node = random.choice(neighbors)

            walk.append(next_node)

        return walk

    def _generate_walks(self, G: nx.Graph) -> List[List[int]]:
        """Generate random walks for all nodes."""
        walks = []
        nodes = list(G.nodes())

        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self._random_walk(G, node)
                walks.append(walk)

        return walks

    def _skip_gram(self, walks: List[List[int]]) -> List[Tuple[int, int]]:
        """Generate training pairs using Skip-gram with sliding window."""
        pairs = []

        for walk in walks:
            for i in range(len(walk)):
                for j in range(max(0, i - self.window_size),
                               min(len(walk), i + self.window_size + 1)):
                    if i != j:
                        pairs.append((walk[i], walk[j]))

        return pairs

    def _negative_sampling(self, G: nx.Graph, batch_size: int) -> List[int]:
        """Generate negative samples based on node degree distribution."""
        degrees = np.array([G.degree(node) for node in G.nodes()])
        probs = degrees ** 0.75  # Raised to 0.75 as in word2vec
        probs = probs / probs.sum()

        return np.random.choice(list(G.nodes()), size=batch_size, p=probs)

    def _sigmoid(self, x: float) -> float:
        """Compute sigmoid function."""
        return 1 / (1 + np.exp(-x))

    def fit(self, G: nx.Graph) -> np.ndarray:
        """Main training procedure."""
        num_nodes = G.number_of_nodes()

        # Initialize embeddings
        self.embeddings = np.random.normal(
            scale=0.01,
            size=(num_nodes, self.embedding_dim)
        )

        # Generate random walks
        walks = self._generate_walks(G)

        # Generate training pairs
        training_pairs = self._skip_gram(walks)

        # Training loop
        for iteration in range(self.max_iter):
            random.shuffle(training_pairs)

            for v_i, v_j in training_pairs:
                # Positive sample update
                z = np.dot(self.embeddings[v_i], self.embeddings[v_j])
                gradient = self._sigmoid(z) - 1

                # Update embeddings for positive sample
                self.embeddings[v_i] -= self.learning_rate * gradient * self.embeddings[v_j]
                self.embeddings[v_j] -= self.learning_rate * gradient * self.embeddings[v_i]

                # Negative sampling
                neg_samples = self._negative_sampling(G, self.neg_batch_size)

                # Update embeddings for negative samples
                for v_k in neg_samples:
                    z = np.dot(self.embeddings[v_i], self.embeddings[v_k])
                    gradient = self._sigmoid(z)

                    self.embeddings[v_i] -= self.learning_rate * gradient * self.embeddings[v_k]
                    self.embeddings[v_k] -= self.learning_rate * gradient * self.embeddings[v_i]

            # Decrease learning rate
            self.learning_rate *= 0.9

        return self.embeddings