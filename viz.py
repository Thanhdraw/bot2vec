import networkx as nx
import numpy as np
from main import Bot2vec

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def visualize_walk(G, walk, title="Random Walk Visualization", filename=None):
    """
    Visualize a single random walk on the graph and save to file
    """
    plt.figure(figsize=(12, 8))

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)

    # Draw all nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightgray',
                           node_size=500)

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2)

    # Draw walk nodes with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(walk)))
    for i in range(len(walk) - 1):
        # Draw node
        nx.draw_networkx_nodes(G, pos,
                               nodelist=[walk[i]],
                               node_color=[colors[i]],
                               node_size=500)
        # Draw edge in walk
        nx.draw_networkx_edges(G, pos,
                               edgelist=[(walk[i], walk[i + 1])],
                               edge_color=colors[i],
                               width=2)

    # Draw last node
    nx.draw_networkx_nodes(G, pos,
                           nodelist=[walk[-1]],
                           node_color=[colors[-1]],
                           node_size=500)

    # Add node labels
    nx.draw_networkx_labels(G, pos)

    plt.title(title)
    plt.axis('off')

    # Save the plot instead of showing it
    if filename:
        plt.savefig(filename)
    plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    # Create sample graph
    G = nx.karate_club_graph()

    # Initialize and run Bot2vec
    model = Bot2vec(G=G, walk_length=10, num_walks=5, p=1.0, q=1.0, r=0.5)
    walks = model.simulate_walks()

    # Visualize the first 3 walks
    for i, walk in enumerate(walks[:3]):
        print(f"Walk {i + 1}:", walk)
        visualize_walk(G, walk, f"Random Walk {i + 1}", f"walk_{i + 1}.png")