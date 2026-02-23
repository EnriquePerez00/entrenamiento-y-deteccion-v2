import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import ssl

# Inject proper certificates globally for PyTorch Hub downloads (MacOS Fix)
import certifi
import ssl
os.environ['SSL_CERT_FILE'] = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

class FeatureExtractor:
    def __init__(self, model_name='mobilenet_v3_small'):
        # Load a pre-trained model for feature extraction
        if model_name == 'mobilenet_v3_small':
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            # Remove the classifier head to get the pooling layer output
            self.model = base_model.features
            self.avgpool = base_model.avgpool
            self.feature_dim = 576 # Standard for MobileNetV3 Small features
        elif model_name == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.model = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 2048
        
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Standard ImageNet transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def get_embedding(self, image):
        """Image can be a PIL Image or a path."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        img_tTask = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(img_tTask)
            if hasattr(self, 'avgpool'):
                features = self.avgpool(features)
            features = torch.flatten(features, 1)
            
        return features.cpu().numpy().flatten()

    def get_batch_embeddings(self, images):
        """Batch processing for speed."""
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            processed_images.append(self.transform(img))
            
        batch_tTask = torch.stack(processed_images).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch_tTask)
            if hasattr(self, 'avgpool'):
                features = self.avgpool(features)
            features = torch.flatten(features, 1)
            
        return features.cpu().numpy()
