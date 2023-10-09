import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch


# Define the dimensions
num_embeddings = 100
embedding_size = 128

# Generate random embeddings
embeddings = np.random.rand(num_embeddings, embedding_size)

def plot_tsne(embeddings, item_dict):
    # Create a t-SNE model with desired parameters
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=0)
    # Fit the model to your embeddings
    tsne_results = tsne.fit_transform(embeddings)
    # Extract the x and y coordinates from the t-SNE results
    x = tsne_results[:, 0]
    y = tsne_results[:, 1]
    # Create a scatter plot
    plt.figure(figsize=(80, 60))
    plt.scatter(x, y, marker='o', color='red', s=10)  # You can customize marker style and size
    # Optionally, add labels or annotations to the points
    for idx in range(0, embedding_matrix.shape[0]):
        print(idx, item_dict[idx])
        plt.annotate(item_dict[idx], (x[idx], y[idx]), fontsize=50)
    # Set plot title and labels
    plt.title("t-SNE Projection of Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    # Show the plot
    plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')


import numpy as np

word_embeddings = {
    "car": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "truck": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "house": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1],
    "tree": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2],
    "plane": [0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3],
    "ship": [0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4],
    "road": [0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5],
    "river": [0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "bird": [0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "wind": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
}

# Convert word embeddings to a 2D NumPy array
embedding_matrix = np.array(list(word_embeddings.values()))

item_dict = dict()
item_keys = list(word_embeddings)
for idx in range(0, embedding_matrix.shape[0]):
    item_dict[idx] = item_keys[idx]

plot_tsne(embedding_matrix, item_dict)

