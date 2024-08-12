import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from torch.utils.data import Dataset

class Config:
    model_path = "../models/model.pth"
    batch_size = 20
    num_epochs = 20
    num_folds = 5
    fold = 0
    image_size = 128
    learning_rate = 1e-4
    weight_decay = 0.05
    num_targets = 6
    target_names = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
    log_targets = ['X18_mean', 'X26_mean', 'X3112_mean']

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
    def __init__(self, num_predictions, num_features, model_name='mobilenet'):
        super(PlantCNN, self).__init__()

        if model_name == 'mobilenet':
            self.cnn = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).features
            self.post_pooling = 1280
        elif model_name == 'efficientnet':
            self.cnn = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).features
            self.post_pooling = 1280
        else:
            self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
            self.post_pooling = 512

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_dropout = nn.Dropout(0.6)
        self.cnn_dense = nn.Linear(self.post_pooling, 128)

        self.dense = nn.Linear(num_features, 64)
        self.feats_dropout = nn.Dropout(0.5)

        self.concat_dense = nn.Linear(128 + 64, 64)
        self.output_layer = nn.Linear(64, num_predictions)

    def forward(self, img, feats):
        # Run CNN
        img = self.cnn(img)
        img = self.global_pooling(img)
        img = self.cnn_dropout(img)
        img = torch.flatten(img, 1)
        img = F.relu(self.cnn_dense(img))

        # Run plant data network
        feats = F.relu(self.dense(feats))
        feats = self.feats_dropout(feats)
        
        # Concatenate data
        combined_data = torch.cat([img, feats], dim=1)
        combined_data = F.relu(self.concat_dense(combined_data))
        output = self.output_layer(combined_data)
        return output
