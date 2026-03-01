"""
Optimize Edilmiş Veri Yükleme Modülleri
Class imbalance çözümü ile
"""

import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, List, Optional
import numpy as np
from collections import Counter

# Orijinal CancerDataset'i import et
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from src.data_loader import CancerDataset
except ImportError:
    # Alternatif import yolu
    import importlib.util
    spec = importlib.util.spec_from_file_location("data_loader", 
                                                   os.path.join(parent_dir, "src", "data_loader.py"))
    data_loader_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_loader_module)
    CancerDataset = data_loader_module.CancerDataset


class CancerDatasetOptimized(CancerDataset):
    """Optimize edilmiş cilt hastalığı veri seti (class imbalance için)"""
    
    def __init__(self, csv_path: str, images_dir: str, transform: Optional[transforms.Compose] = None, 
                 train: bool = True, train_split: float = 0.8):
        super().__init__(csv_path, images_dir, transform, train, train_split)
    
    def get_class_weights(self):
        """Sınıf ağırlıklarını hesapla (class imbalance için)"""
        labels = [item['label'] for item in self.data]
        class_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = len(self.CLASS_NAMES)
        
        # Her sınıf için ağırlık hesapla
        weights = []
        for i in range(num_classes):
            count = class_counts.get(i, 1)  # Eğer sınıf yoksa 1 kullan
            weight = total_samples / (num_classes * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_sampler(self):
        """WeightedRandomSampler oluştur"""
        labels = [item['label'] for item in self.data]
        class_weights = self.get_class_weights()
        
        # Her örnek için ağırlık
        sample_weights = [class_weights[label] for label in labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )


def get_cancer_transforms_optimized(train: bool = True):
    """Optimize edilmiş transformasyonlar"""
    if train:
        # Güçlü augmentation pipeline
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/Inference için
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_weighted_dataloader(dataset, batch_size: int = 32, num_workers: int = 0):
    """WeightedRandomSampler ile DataLoader oluştur"""
    sampler = dataset.get_sampler()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

