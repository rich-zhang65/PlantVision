import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights, mobilenet_v2, MobileNet_V2_Weights
from PIL import Image
from torch.utils.data import Dataset

class Config:
    model_path = "../models/model.pth"
    batch_size = 20
    num_epochs = 5
    num_folds = 5
    fold = 0
    image_size = 128
    learning_rate = 1e-3
    weight_decay = 1e-4
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
    def __init__(self, num_predictions, num_features, model_name='resnet'):
        super(PlantCNN, self).__init__()

        if model_name == 'mobilenet':
            self.cnn = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        else:
            self.cnn = resnet34(weights=ResNet34_Weights.DEFAULT)
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])
        # self.cnn_dropout = nn.Dropout(0.2)

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_dense1 = nn.Linear(512, 512)
        self.cnn_dense2 = nn.Linear(512, 1)

        self.dense1 = nn.Linear(num_features, 64)
        self.dense2 = nn.Linear(64, 32)
        self.dense3 = nn.Linear(32, 4)
        # self.feat_dropout = nn.Dropout(0.1)

        self.concat_dense1 = nn.Linear(1 + 4, 8)
        self.concat_dense2 = nn.Linear(8, 8)
        self.concat_dense3 = nn.Linear(8, 4)
        self.output_layer = nn.Linear(4, num_predictions)
        # self.output_layer = nn.Linear(1064, num_predictions)

    def forward(self, img, feats):
        # Run CNN
        img = self.cnn(img)
        img = self.global_pooling(img)
        img = torch.flatten(img, 1)
        img = F.relu(self.cnn_dense1(img))
        img = self.cnn_dense2(img)

        # Run plant data network
        feats = F.relu(self.dense1(feats))
        feats = F.relu(self.dense2(feats))
        feats = self.dense3(feats)
        
        # Concatenate data
        combined_data = torch.cat([img, feats], dim=1)
        combined_data = F.relu(self.concat_dense1(combined_data))
        combined_data = F.relu(self.concat_dense2(combined_data))
        combined_data = F.relu(self.concat_dense3(combined_data))
        output = self.output_layer(combined_data)
        return output
