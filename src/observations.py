import matplotlib.pyplot as plt
from plants import Config

def visualize_data(train_dataloader):
    inps, labels = next(iter(train_dataloader))
    imgs = inps['images']
    feats = inps['features']
    num_imgs, num_cols = 8, 4

    plt.figure(figsize=(4 * num_cols, num_imgs // num_cols * 5))

    for i, (img, tar) in enumerate(zip(imgs[:num_imgs], labels[:num_imgs])):
        plt.subplot(num_imgs // num_cols, num_cols, i + 1)

        # Normalize the image to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-4)

        formatted_tar = "\n".join(
            [
                ", ".join(
                    f"{name.replace('_mean','')}: {val:.2f}"
                    for name, val in zip(Config.target_names[j : j + 3], tar[j : j + 3])
                )
                for j in range(0, len(Config.target_names), 3)
            ]
        )

        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(f"[{formatted_tar}]")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def display_target_dist(labels):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.hist(labels[:, i], bins=128)
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
