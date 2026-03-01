"""
Cilt Tipi Analizi Modeli
PyTorch tabanlı CNN model wrapper
"""

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, Optional
import os


class SkinTypeCNN(nn.Module):
    """Cilt tipi sınıflandırması için CNN modeli"""
    
    def __init__(self, num_classes: int = 3):
        super(SkinTypeCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class SkinTypeModel:
    """Cilt tipi analizi için model wrapper"""
    
    SKIN_TYPES = {
        0: {
            "name": "Kuru",
            "description": "Cildiniz yeterli sebum üretmiyor. Nem kaybı ve pullanma görülebilir."
        },
        1: {
            "name": "Normal",
            "description": "Cildiniz dengeli. Sebum üretimi ve nem seviyesi optimal."
        },
        2: {
            "name": "Yağlı",
            "description": "Cildiniz fazla sebum üretiyor. Parlaklık ve gözenek sorunları görülebilir."
        }
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SkinTypeCNN(num_classes=3)
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
        """Görüntüden cilt tipi tahmini yap"""
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
            skin_type_info = self.SKIN_TYPES.get(predicted_class, {
                "name": "Belirsiz",
                "description": "Model cilt tipini belirleyemedi."
            })
            
            return {
                "skin_type": skin_type_info["name"],
                "confidence": round(confidence * 100, 2),
                "description": skin_type_info["description"],
                "all_probabilities": {
                    self.SKIN_TYPES[i]["name"]: round(probabilities[0][i].item() * 100, 2)
                    for i in range(len(self.SKIN_TYPES))
                }
            }
        except Exception as e:
            return {
                "error": f"Tahmin yapılırken hata oluştu: {str(e)}",
                "skin_type": "Hata",
                "confidence": 0.0,
                "description": "Lütfen geçerli bir görüntü yükleyin."
            }

