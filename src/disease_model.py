"""
Cilt Hastalığı Tespiti Modeli
PyTorch tabanlı CNN model wrapper
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, Tuple, Optional
import os


class SkinDiseaseCNN(nn.Module):
    """Basit CNN modeli cilt hastalığı sınıflandırması için"""
    
    def __init__(self, num_classes: int = 7):
        super(SkinDiseaseCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 256 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class DiseaseModel:
    """Cilt hastalığı tespiti için model wrapper"""
    
    # Hastalık sınıfları ve açıklamaları
    DISEASE_CLASSES = {
        0: {
            "name": "Aktinik Keratoz",
            "risk": "Düşük-Orta",
            "description": "Güneş hasarına bağlı gelişen, genellikle iyi huylu lezyonlar. Düzenli takip önerilir."
        },
        1: {
            "name": "Bazal Hücreli Karsinom",
            "risk": "Yüksek",
            "description": "En yaygın cilt kanseri türü. Erken teşhis ve tedavi çok önemlidir. Mutlaka dermatoloğa başvurun."
        },
        2: {
            "name": "Benign Keratoz",
            "risk": "Düşük",
            "description": "İyi huylu lezyon. Genellikle tedavi gerektirmez ancak değişiklik gösterirse kontrol edilmelidir."
        },
        3: {
            "name": "Dermatofibrom",
            "risk": "Düşük",
            "description": "İyi huylu, genellikle zararsız lezyon. Estetik kaygı varsa dermatoloğa danışılabilir."
        },
        4: {
            "name": "Melanom",
            "risk": "Çok Yüksek",
            "description": "En tehlikeli cilt kanseri türü. Acil dermatoloji konsültasyonu gereklidir."
        },
        5: {
            "name": "Melanositik Nevüs (Ben)",
            "risk": "Düşük",
            "description": "Genellikle zararsız ben. Ancak şekil, renk veya boyut değişikliği durumunda kontrol edilmelidir."
        },
        6: {
            "name": "Vasküler Lezyon",
            "risk": "Düşük",
            "description": "Kan damarlarıyla ilgili lezyonlar. Çoğunlukla zararsızdır ancak büyüme gösterirse değerlendirilmelidir."
        }
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SkinDiseaseCNN(num_classes=7)
        self.model.eval()
        
        # Transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Model yükleme
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model başarıyla yüklendi: {model_path}")
            except Exception as e:
                print(f"Model yüklenirken hata: {e}. Varsayılan ağırlıklar kullanılıyor.")
        else:
            print("Model dosyası bulunamadı. Varsayılan ağırlıklar kullanılıyor.")
        
        self.model.to(self.device)
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Görüntüyü model için hazırla"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def predict(self, image: Image.Image) -> Dict:
        """Görüntüden hastalık tahmini yap"""
        try:
            # Ön işleme
            image_tensor = self.preprocess_image(image)
            
            # Tahmin
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Sonuç hazırlama
            disease_info = self.DISEASE_CLASSES.get(predicted_class, {
                "name": "Bilinmeyen",
                "risk": "Belirsiz",
                "description": "Model bu lezyonu sınıflandıramadı."
            })
            
            return {
                "disease_class": disease_info["name"],
                "risk_level": disease_info["risk"],
                "confidence": round(confidence * 100, 2),
                "description": disease_info["description"],
                "all_probabilities": {
                    self.DISEASE_CLASSES[i]["name"]: round(probabilities[0][i].item() * 100, 2)
                    for i in range(len(self.DISEASE_CLASSES))
                }
            }
        except Exception as e:
            return {
                "error": f"Tahmin yapılırken hata oluştu: {str(e)}",
                "disease_class": "Hata",
                "risk_level": "Belirsiz",
                "confidence": 0.0,
                "description": "Lütfen geçerli bir görüntü yükleyin."
            }

