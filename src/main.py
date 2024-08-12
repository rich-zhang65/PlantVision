import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

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

def get_normalized_labels(data, train=False):
    normalized = np.zeros_like(data[Config.target_names], dtype=np.float64)
    means = np.zeros(Config.num_targets)
    stds = np.zeros(Config.num_targets)

    for idx, target in enumerate(Config.target_names):
        normalized[:,idx] = data[target]
        
        if target in Config.log_targets:
            normalized[:,idx] = np.log10(normalized[:,idx])
        
        means[idx] = np.mean(normalized[:,idx])
        normalized[:,idx] = normalized[:,idx] - np.median(normalized[:,idx])
        
        stds[idx] = np.std(normalized[:,idx])
        normalized[:,idx] = normalized[:,idx] / np.std(normalized[:,idx])
    
    if train:
        return normalized, means, stds
    else:
        return normalized

def denormalize_labels(data, means, stds):
    log_features = np.isin(Config.target_names, Config.log_targets)
    denormalized = data * stds + means
    denormalized[:, log_features] = 10 ** denormalized[:, log_features]
    return denormalized

def create_folds(train_data):
    new_data = train_data.copy()
    new_data['final_bin'] = ''
    skf = StratifiedKFold(n_splits=Config.num_folds, shuffle=True)

    for i, target in enumerate(Config.target_names):
        bin_edges = np.percentile(new_data[target], np.linspace(0, 100, Config.num_folds + 1))
        bin = np.digitize(new_data[target], bin_edges)
        bin_str = bin.astype(str)
        new_data['final_bin'] += bin_str
    
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

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs=Config.num_epochs):
    best_r2_score = -np.inf
    last_improvement = -1

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

            outputs = model(imgs, feats)

            loss = criterion(outputs, targets)
            # print(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

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
            last_improvement = epoch

        scheduler.step(avg_val_r2)

        if last_improvement + 5 <= epoch:
            break

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

    transform = transforms.Compose([
        transforms.RandomResizedCrop(Config.image_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        Cutout(size=32, p=0.7),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
    ])

    scaler = StandardScaler()

    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')

    FEATURE_COLS = test_data.columns.values[1:]
    num_features = len(FEATURE_COLS)

    # Mask outliers
    for column in Config.target_names:
        lower_quantile = train_data[column].quantile(0.001)
        upper_quantile = train_data[column].quantile(0.98)
        train_data = train_data[(train_data[column] >= lower_quantile) & (train_data[column] <= upper_quantile)]
    train_data = train_data.reset_index(drop=True)

    # Split into training and validation sets
    new_data = create_folds(train_data)
    train_data = new_data[new_data.fold != Config.fold]
    valid_data = new_data[new_data.fold == Config.fold]

    # Validate fold distributions
    # for i in range(Config.num_folds):
    #     fold_data = new_data[new_data.fold == i]
    #     print(len(fold_data))
    #     fold_labels = fold_data[Config.target_names].values
    #     observations.display_target_dist(fold_labels)

    # Training data
    train_paths = np.char.add(np.char.add(TRAINING_IMAGES_PATH, np.vectorize(str)(train_data.id.values)), '.jpeg')
    train_features = train_data[FEATURE_COLS].values
    train_features = scaler.fit_transform(train_features)
    train_labels = train_data[Config.target_names].values
    train_labels, label_means, label_stds = get_normalized_labels(train_data, train=True)

    # Display the distribution of labels
    # observations.display_target_dist(train_labels)

    train_dataset = PlantDataset(train_paths, train_features, train_labels, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=6, pin_memory=True)

    # Visualize training data
    # observations.visualize_data(train_dataloader)

    # Validation data
    valid_paths = np.char.add(np.char.add(TRAINING_IMAGES_PATH, np.vectorize(str)(valid_data.id.values)), '.jpeg')
    valid_features = valid_data[FEATURE_COLS].values
    valid_features = scaler.transform(valid_features)
    valid_labels = valid_data[Config.target_names].values
    valid_labels = get_normalized_labels(valid_data)

    valid_dataset = PlantDataset(valid_paths, valid_features, valid_labels)
    valid_dataloader = DataLoader(valid_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=6, pin_memory=True)

    # Testing data
    test_paths = np.char.add(np.char.add(TEST_IMAGES_PATH, np.vectorize(str)(test_data.id.values)), '.jpeg')
    test_features = test_data[FEATURE_COLS].values
    test_features = scaler.transform(test_features)

    test_dataset = PlantDataset(test_paths, test_features)
    test_dataloader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=6)

    # Train and test model
    model = PlantCNN(Config.num_targets, num_features)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.2, patience=2, verbose=True)
    criterion = torch.nn.SmoothL1Loss()

    train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, scheduler, Config.num_epochs)
    model.load_state_dict(torch.load(Config.model_path))
    evaluation = evaluate_model(model, test_dataloader)
    prediction = denormalize_labels(evaluation, label_means, label_stds)

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
