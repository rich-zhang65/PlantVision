import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from plants import Config, PlantDataset, PlantCNN
import observations

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == torch.device("cuda"):
    print('Utilizing CUDA')
else:
    print('Not utilizing CUDA')

class Cutout(object):
    def __init__(self, size, p=0.5):
        self.size = size
        self.p = p

    def __call__(self, img):
        if torch.rand(1) < self.p:
            img = transforms.functional.pil_to_tensor(img)
            _, h, w = img.shape
            y = torch.randint(0, h, (1,))
            x = torch.randint(0, w, (1,))

            y1 = torch.clamp(y - self.size // 2, 0, h).item()
            y2 = torch.clamp(y + self.size // 2, 0, h).item()
            x1 = torch.clamp(x - self.size // 2, 0, w).item()
            x2 = torch.clamp(x + self.size // 2, 0, w).item()

            img[:, y1:y2, x1:x2] = 0
            img = transforms.functional.to_pil_image(img)

        return img

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

def r2_score(y_true, y_pred):
    SS_res = np.sum((y_true - y_pred)**2, axis=0)
    SS_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
    r2 = 1 - SS_res / (SS_tot + 1e-6)
    mean_r2 = np.mean(r2)
    return mean_r2

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=Config.num_epochs):
    model.train()
    for epoch in range(num_epochs):
        # model.train()
        total_r2_train = 0.0
        trained_batches = 0
        min_r2 = np.inf
        max_r2 = -np.inf

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
            imgs = inputs['images'].to(device, dtype=torch.float32)
            feats = inputs['features'].to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.float32)

            optimizer.zero_grad()

            outputs = model(imgs, feats)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            targets_np = targets.cpu().detach().numpy()
            outputs_np = outputs.cpu().detach().numpy()
            # print(targets_np, outputs_np)

            r2_value = r2_score(targets_np, outputs_np)
            # print(f"R2 Value: {r2_value}")

            min_r2 = min(min_r2, r2_value)
            max_r2 = max(max_r2, r2_value)
            total_r2_train += r2_value
            trained_batches += 1

        avg_r2 = total_r2_train / trained_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average R2 Score: {avg_r2}, Min R2: {min_r2}, Max R2: {max_r2}")

        # validation
        # model.eval()
        # total_r2_test = 0.0
        # tested_batches = 0

        # with torch.no_grad():
        #     for batch_idx, inputs

def evaluate_model(model, test_dataloader):
    model.eval()
    predictions = []

    for batch_idx, inputs in enumerate(tqdm(test_dataloader, desc='Testing')):
        imgs = inputs['images'].to(device, dtype=torch.float32)
        feats = inputs['features'].to(device, dtype=torch.float32)

        with torch.no_grad():
            outputs = model(imgs, feats)
        
        predictions.append(outputs.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    return predictions

def main():
    TRAINING_IMAGES_PATH = "../data/train_images/"
    TEST_IMAGES_PATH = "../data/test_images/"

    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')

    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        Cutout(size=32, p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    scaler = StandardScaler()

    train_paths = np.char.add(np.char.add(TRAINING_IMAGES_PATH, np.vectorize(str)(train_data.id.values)), '.jpeg')
    train_features = scaler.fit_transform(train_data[test_data.columns[1:].tolist()].values)
    labels = train_data[Config.target_names].values

    num_features = len(train_features[0])

    # Display the distribution of labels
    observations.display_target_dist(labels)

    train_dataset = PlantDataset(train_paths, train_features, labels, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=6, drop_last=True, pin_memory=True)

    # Visualize training data
    visualize_data(train_dataloader)

    test_paths = np.char.add(np.char.add(TEST_IMAGES_PATH, np.vectorize(str)(test_data.id.values)), '.jpeg')
    test_features = scaler.transform(test_data[test_data.columns[1:].tolist()].values)

    test_dataset = PlantDataset(test_paths, test_features)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=6)

    model = PlantCNN(Config.num_targets, num_features)
    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=1e-5)
    criterion = torch.nn.HuberLoss()

    train_model(model, train_dataloader, None, criterion, optimizer, Config.num_epochs)
    predictions = evaluate_model(model, test_dataloader)

    submission = test_data[['id']].copy()
    target_cols = [ name.replace('_mean', '') for name in Config.target_names ]
    submission[target_cols] = predictions.tolist()

    submission.to_csv('../results/submission.csv', index=False)

if __name__ == '__main__':
    main()
