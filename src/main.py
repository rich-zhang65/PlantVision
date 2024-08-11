import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

from plants import Config, PlantDataset, PlantCNN
import observations

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def normalize_data(data, train=False):
    log_data = np.log10(np.maximum(data, 1e-6))

    # remove outliers

    min_values = np.min(log_data, axis=0)
    max_values = np.max(log_data, axis=0)
    normalized = (log_data - min_values) / (max_values - min_values)

    if train:
        return normalized, min_values, max_values
    else:
        return normalized

def denormalize_data(data, min_values, max_values):
    # print("Data:", data)
    log_values = data * (max_values - min_values) + min_values
    # print("Log values:", log_values)
    denormalized = np.power(10, log_values)
    # print("Denormalized:", denormalized)
    return denormalized

def create_folds(train_data):
    new_data = train_data.copy()
    new_data['final_bin'] = ''
    skf = StratifiedKFold(n_splits=Config.num_folds, shuffle=True)

    for i, target in enumerate(Config.target_names):
        bin_edges = np.percentile(new_data[target], np.linspace(0, 100, Config.num_folds + 1))
        new_data['final_bin'] = new_data['final_bin'] + str(np.digitize(new_data[target], bin_edges))
    
    new_data['fold'] = -1

    for fold, (train_idx, valid_idx) in enumerate(skf.split(new_data, new_data['final_bin'])):
        new_data.loc[valid_idx, 'fold'] = fold
    return new_data

def r2_score(y_true, y_pred):
    SS_res = np.sum((y_true - y_pred)**2, axis=0)
    SS_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
    r2 = 1 - SS_res / (SS_tot + 1e-6)
    mean_r2 = np.mean(r2)
    return mean_r2

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=Config.num_epochs):
    best_r2_score = -np.inf

    for epoch in range(num_epochs):
        model.train()
        total_r2_train = 0.0
        trained_batches = 0
        min_train_r2 = np.inf
        max_train_r2 = -np.inf

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader, desc=f'Training Epoch {epoch + 1}/{num_epochs}')):
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

            r2_value = r2_score(targets_np, outputs_np)

            min_train_r2 = min(min_train_r2, r2_value)
            max_train_r2 = max(max_train_r2, r2_value)
            total_r2_train += r2_value
            trained_batches += 1

        avg_train_r2 = total_r2_train / trained_batches
        print(f"Training Epoch {epoch + 1}/{num_epochs}, Average R2 Score: {avg_train_r2}, Min R2: {min_train_r2}, Max R2: {max_train_r2}")

        model.eval()
        total_r2_val = 0.0
        validated_batches = 0
        min_val_r2 = np.inf
        max_val_r2 = -np.inf

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(val_dataloader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}')):
                imgs = inputs['images'].to(device, dtype=torch.float32)
                feats = inputs['features'].to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.float32)

                outputs = model(imgs, feats)
                print(outputs)

                targets_np = targets.cpu().detach().numpy()
                outputs_np = outputs.cpu().detach().numpy()

                r2_value = r2_score(targets_np, outputs_np)

                min_val_r2 = min(min_val_r2, r2_value)
                max_val_r2 = max(max_val_r2, r2_value)
                total_r2_val += r2_value
                validated_batches += 1
        
        avg_val_r2 = total_r2_val / validated_batches
        print(f"Validation Epoch {epoch + 1}/{num_epochs}, Average R2 Score: {avg_val_r2}, Min R2: {min_val_r2}, Max R2: {max_val_r2}")

        if avg_val_r2 > best_r2_score:
            best_r2_score = avg_val_r2
            torch.save(model.state_dict(), Config.model_path)

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
    ])

    scaler = StandardScaler()

    new_data = create_folds(train_data)
    train_data = new_data[new_data.fold != Config.fold]
    valid_data = new_data[new_data.fold == Config.fold]

    train_paths = np.char.add(np.char.add(TRAINING_IMAGES_PATH, np.vectorize(str)(train_data.id.values)), '.jpeg')
    train_features = train_data[test_data.columns[1:].tolist()].values
    train_features = normalize_data(train_features)
    train_features = scaler.fit_transform(train_features)
    train_labels = train_data[Config.target_names].values
    train_labels, train_min, train_max = normalize_data(train_labels, train=True)

    num_features = len(train_features[0])

    # Display the distribution of labels
    observations.display_target_dist(train_labels)

    train_dataset = PlantDataset(train_paths, train_features, train_labels, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=6, pin_memory=True)

    # Visualize training data
    # visualize_data(train_dataloader)

    valid_paths = np.char.add(np.char.add(TRAINING_IMAGES_PATH, np.vectorize(str)(valid_data.id.values)), '.jpeg')
    valid_features = valid_data[test_data.columns[1:].tolist()].values
    valid_features = normalize_data(valid_features)
    valid_features = scaler.transform(valid_features)
    valid_labels = valid_data[Config.target_names].values
    valid_labels = normalize_data(valid_labels)

    valid_dataset = PlantDataset(valid_paths, valid_features, valid_labels)
    valid_dataloader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=6, pin_memory=True)

    test_paths = np.char.add(np.char.add(TEST_IMAGES_PATH, np.vectorize(str)(test_data.id.values)), '.jpeg')
    test_features = test_data[test_data.columns[1:].tolist()].values
    test_features = normalize_data(test_features)
    test_features = scaler.transform(test_features)

    test_dataset = PlantDataset(test_paths, test_features)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=6)

    model = PlantCNN(Config.num_targets, num_features)
    model.to(device)
    # print(model)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    criterion = torch.nn.MSELoss()

    train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, Config.num_epochs)
    model.load_state_dict(torch.load(Config.model_path))
    evaluation = evaluate_model(model, test_dataloader)
    prediction = denormalize_data(evaluation, train_min, train_max)

    submission = test_data[['id']].copy()
    target_cols = [ name.replace('_mean', '') for name in Config.target_names ]
    submission[target_cols] = prediction.tolist()

    submission.to_csv('../results/submission.csv', index=False)

if __name__ == '__main__':
    if device == torch.device("cuda"):
        print('Utilizing CUDA')
    else:
        print('Not utilizing CUDA')

    main()
