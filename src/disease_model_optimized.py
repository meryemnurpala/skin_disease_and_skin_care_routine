"""
Optimize Edilmiş Cilt Hastalığı Tespiti Modeli
EfficientNet tabanlı, pretrained weights ile
"""

import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, Optional
import os


class EfficientNetDiseaseModel(nn.Module):
    """EfficientNet tabanlı cilt hastalığı sınıflandırma modeli"""
    
    def __init__(self, num_classes: int = 7, model_name: str = 'efficientnet_b0'):
        super(EfficientNetDiseaseModel, self).__init__()
        
        # Pretrained EfficientNet yükle
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        elif model_name == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(weights='IMAGENET1K_V1')
        else:
            raise ValueError(f"Desteklenmeyen model: {model_name}")
        
        # Feature extractor'ı freeze et (başlangıçta)
        # Fine-tuning sırasında açılacak
        
        # Classifier'ı değiştir
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def freeze_backbone(self):
        """Backbone'u freeze et (feature extractor)"""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Backbone'u unfreeze et (fine-tuning için)"""
        for param in self.backbone.features.parameters():
            param.requires_grad = True
    
    def unfreeze_last_layers(self, num_layers: int = 3):
        """Son birkaç katmanı unfreeze et"""
        # Son num_layers katmanını unfreeze et
        total_blocks = len(list(self.backbone.features))
        for i, block in enumerate(self.backbone.features):
            if i >= total_blocks - num_layers:
                for param in block.parameters():
                    param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)


class OptimizedDiseaseModel:
    """Optimize edilmiş cilt hastalığı tespiti model wrapper"""
    
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
    
    # Minimum güven eşiği
    MIN_CONFIDENCE_THRESHOLD = 0.5
    
    def __init__(self, model_path: Optional[str] = None, model_name: str = 'efficientnet_b0'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EfficientNetDiseaseModel(num_classes=7, model_name=model_name)
        self.model.eval()
        
        # Inference için transform (validation ile aynı)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        ])
        
        # Model yükleme
        if model_path and os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"Model basariyla yuklendi: {model_path}")
            except Exception as e:
                print(f"Model yuklenirken hata: {e}. Varsayilan agirliklar kullaniliyor.")
        else:
            print("Model dosyasi bulunamadi. Varsayilan agirliklar kullaniliyor.")
        
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
                max_prob = confidence
            
            # Güven kontrolü
            is_uncertain = max_prob < self.MIN_CONFIDENCE_THRESHOLD
            
            # Sonuç hazırlama
            disease_info = self.DISEASE_CLASSES.get(predicted_class, {
                "name": "Bilinmeyen",
                "risk": "Belirsiz",
                "description": "Model bu lezyonu siniflandiramadi."
            })
            
            result = {
                "disease_class": disease_info["name"],
                "risk_level": disease_info["risk"],
                "confidence": round(confidence * 100, 2),
                "description": disease_info["description"],
                "all_probabilities": {
                    self.DISEASE_CLASSES[i]["name"]: round(probabilities[0][i].item() * 100, 2)
                    for i in range(len(self.DISEASE_CLASSES))
                },
                "is_uncertain": is_uncertain
            }
            
            # Belirsizlik uyarısı
            if is_uncertain:
                result["warning"] = (
                    f"⚠️ Model tahmininde emin degil (Güven: {confidence*100:.1f}%). "
                    "Lutfen bir dermatologa basvurun."
                )
            
            return result
        except Exception as e:
            return {
                "error": f"Tahmin yapilirken hata olustu: {str(e)}",
                "disease_class": "Hata",
                "risk_level": "Belirsiz",
                "confidence": 0.0,
                "description": "Lutfen gecerli bir goruntu yukleyin.",
                "is_uncertain": True
            }

