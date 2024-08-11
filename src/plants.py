import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from PIL import Image
from torch.utils.data import Dataset

class Config:
    batch_size = 8
    num_epochs = 5
    image_size = 128
    learning_rate = 1e-4
    num_targets = 6
    target_names = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']

class PlantDataset(Dataset):
    def __init__(self, paths, features, labels=None, transform=None):
        self.paths = paths
        self.features = features
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        feature = self.features[idx]

        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float32)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx])
            return { 'images': image, 'features': feature }, label
        else:
            return { 'images': image, 'features': feature }

class PlantCNN(nn.Module):
    def __init__(self, num_predictions, num_features):
        super(PlantCNN, self).__init__()

        self.cnn = resnet34(weights=ResNet34_Weights.DEFAULT)
        self.cnn_dropout = nn.Dropout(0.2)

        self.dense1 = nn.Linear(num_features, num_features * 2)
        self.dense2 = nn.Linear(num_features * 2, 64)
        # self.dense = nn.Linear(num_features, 64)
        self.feat_dropout = nn.Dropout(0.1)

        self.output_layer = nn.Linear(1064, num_predictions)

    def forward(self, img, feats):
        # Run CNN
        img = self.cnn(img)
        img = self.cnn_dropout(torch.flatten(img, 1))

        # Run plant data network
        feats = self.dense1(feats)
        feats = F.relu(feats)
        feats = self.dense2(feats)
        feats = F.relu(feats)
        feats = self.feat_dropout(feats)

        # Concatenate data
        combined_data = torch.cat([img, feats], dim=1)
        output = self.output_layer(combined_data)

        return output
