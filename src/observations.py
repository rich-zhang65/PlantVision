import matplotlib.pyplot as plt
from plants import Config

def display_target_dist(labels):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.hist(labels[:, i], bins=50, edgecolor='black')
        ax.set_title(Config.target_names[i])
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def display_feature_dist(features):
    fig, axes = plt.subplots(10, 17, figsize=(18, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i >= features.shape[1]: break
        
        ax.hist(features[:, i], bins=50, edgecolor='black')
        ax.grid(True)

    plt.tight_layout()
    plt.show()
