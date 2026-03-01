"""
Veri yükleme modülleri
Cancer ve Skin Type veri setleri için
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, List, Optional
import numpy as np


class CancerDataset(Dataset):
    """Cilt hastalığı veri seti"""
    
    # CSV sınıflarından model sınıflarına mapping
    CLASS_MAPPING = {
        'MEL': 4,      # Melanom
        'NV': 5,       # Melanositik Nevüs (Ben)
        'BCC': 1,      # Bazal Hücreli Karsinom
        'AKIEC': 0,    # Aktinik Keratoz
        'BKL': 2,      # Benign Keratoz
        'DF': 3,       # Dermatofibrom
        'VASC': 6      # Vasküler Lezyon
    }
    
    CLASS_NAMES = [
        "Aktinik Keratoz",
        "Bazal Hücreli Karsinom",
        "Benign Keratoz",
        "Dermatofibrom",
        "Melanom",
        "Melanositik Nevüs (Ben)",
        "Vasküler Lezyon"
    ]
    
    def __init__(self, csv_path: str, images_dir: str, transform: Optional[transforms.Compose] = None, 
                 train: bool = True, train_split: float = 0.8):
        """
        Args:
            csv_path: GroundTruth.csv dosya yolu
            images_dir: Görüntülerin bulunduğu klasör
            transform: Görüntü transformasyonları
            train: True ise train seti, False ise validation seti
            train_split: Train/validation split oranı
        """
        self.images_dir = images_dir
        self.transform = transform
        
        # CSV dosyasını oku
        df = pd.read_csv(csv_path)
        
        # Sınıf kolonlarını al
        class_columns = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        
        # Her görüntü için sınıf etiketini belirle
        self.data = []
        for idx, row in df.iterrows():
            image_name = row['image']
            image_path = os.path.join(images_dir, f"{image_name}.jpg")
            
            # Dosya var mı kontrol et
            if not os.path.exists(image_path):
                continue
            
            # One-hot encoding'den sınıf indeksine çevir
            class_probs = row[class_columns].values
            class_idx = np.argmax(class_probs)
            
            # Mapping ile model sınıfına çevir
            csv_class = class_columns[class_idx]
            model_class = self.CLASS_MAPPING[csv_class]
            
            self.data.append({
                'image_path': image_path,
                'label': model_class,
                'image_name': image_name
            })
        
        # Train/validation split
        total_samples = len(self.data)
        split_idx = int(total_samples * train_split)
        
        if train:
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]
        
        print(f"{'Train' if train else 'Validation'} seti: {len(self.data)} örnek")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Görüntüyü yükle
        image = Image.open(item['image_path']).convert('RGB')
        label = item['label']
        
        # Transform uygula
        if self.transform:
            image = self.transform(image)
        
        return image, label


class SkinTypeDataset(Dataset):
    """Cilt tipi veri seti"""
    
    CLASS_MAPPING = {
        'dry': 0,
        'normal': 1,
        'oily': 2
    }
    
    CLASS_NAMES = ['Kuru', 'Normal', 'Yağlı']
    
    def __init__(self, data_dir: str, split: str = 'train', 
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            data_dir: skin_type klasörünün yolu (train/valid/test içeren)
            split: 'train', 'valid', veya 'test'
            transform: Görüntü transformasyonları
        """
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.data = []
        
        # Her sınıf klasöründen görüntüleri topla
        for class_name, class_idx in self.CLASS_MAPPING.items():
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Uyarı: {class_dir} bulunamadı")
                continue
            
            # Klasördeki tüm görüntüleri al
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(class_dir, filename)
                    self.data.append({
                        'image_path': image_path,
                        'label': class_idx,
                        'class_name': class_name
                    })
        
        print(f"{split.capitalize()} seti: {len(self.data)} örnek")
        for class_name, class_idx in self.CLASS_MAPPING.items():
            count = sum(1 for item in self.data if item['label'] == class_idx)
            print(f"  - {class_name}: {count} örnek")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Görüntüyü yükle
        image = Image.open(item['image_path']).convert('RGB')
        label = item['label']
        
        # Transform uygula
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_cancer_transforms(train: bool = True):
    """Cilt hastalığı veri seti için transformasyonlar (Optimize edilmiş)"""
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
        # Validation/Inference için sadece resize, center crop, normalize
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def get_skin_type_transforms(train: bool = True):
    """Cilt tipi veri seti için transformasyonlar"""
    if train:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

