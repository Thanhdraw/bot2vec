import networkx as nx
import numpy as np
from community import community_louvain
from typing import List, Dict, Set


class Bot2vec:
    def __init__(self, G: nx.Graph, walk_length: int, num_walks: int, p: float, q: float, r: float):
        """
        Initialize Bot2vec algorithm_1

        Args:
            G: Input graph
            walk_length: Length of each random walk
            num_walks: Number of walks per node
            p, q, r: Model parameters
        """
        self.G = G
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.r = r
        self.walks = []

    def detect_communities(self) -> Dict:
        """Detect communities using Louvain method"""
        return community_louvain.best_partition(self.G)

    def compute_transition_prob(self, v_curr: int, v_next: int, communities: Dict) -> float:
        """
        Compute transition probability between nodes
        """
        # If nodes are in same community
        if communities[v_curr] == communities[v_next]:
            return self.compute_alpha(v_curr, v_next)
        else:
            return self.compute_beta(v_curr, v_next)

    def compute_alpha(self, v_curr: int, v_next: int) -> float:
        """Compute α (alpha) transition probability for same community"""
        return 1.0 / len(list(self.G.neighbors(v_curr)))

    def compute_beta(self, v_curr: int, v_next: int) -> float:
        """Compute β (beta) transition probability for different communities"""
        return (1.0 / len(list(self.G.neighbors(v_curr)))) * self.r

    def alias_setup(self, probs: List[float]) -> tuple:
        """
        Create alias table for efficient sampling
        """
        K = len(probs)
        q = np.zeros(K)
        # Sửa np.int thành np.int64
        J = np.zeros(K, dtype=np.int64)

        # Initialize
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Process
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] - (1.0 - q[small])

            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    def alias_draw(self, J: np.ndarray, q: np.ndarray) -> int:
        """Draw sample from alias table"""
        K = len(J)
        kk = int(np.floor(np.random.rand() * K))

        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    def do_walk(self, start_node: int, communities: Dict) -> List[int]:
        """
        Perform a single random walk
        """
        walk = [start_node]

        for i in range(self.walk_length - 1):
            cur = walk[-1]
            cur_nbrs = list(self.G.neighbors(cur))

            if len(cur_nbrs) > 0:
                # Calculate transition probabilities
                probs = []
                for nbr in cur_nbrs:
                    prob = self.compute_transition_prob(cur, nbr, communities)
                    probs.append(prob)

                # Normalize probabilities
                probs = [p / sum(probs) for p in probs]

                # Set up alias sampling
                J, q = self.alias_setup(probs)

                # Sample next node
                next_idx = self.alias_draw(J, q)
                next_node = cur_nbrs[next_idx]

                walk.append(next_node)
            else:
                break

        return walk

    def simulate_walks(self) -> List[List[int]]:
        """
        Simulate all random walks
        """
        # Detect communities
        communities = self.detect_communities()

        # Generate walks
        walks = []
        nodes = list(self.G.nodes())

        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self.do_walk(node, communities)
                walks.append(walk)

        return walks


# Example usage
if __name__ == "__main__":
    # Create a sample graph
    G = nx.karate_club_graph()

    # Initialize Bot2vec
    model = Bot2vec(
        G=G,
        walk_length=10,
        num_walks=5,
        p=1.0,
        q=1.0,
        r=0.5
    )

    # Generate walks
    walks = model.simulate_walks()

    # Print first few walks
    print("Sample walks:")
    for walk in walks[:3]:
        print(walk)