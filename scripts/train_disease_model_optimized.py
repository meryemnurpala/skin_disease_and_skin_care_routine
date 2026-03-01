"""
Optimize Edilmiş Cilt Hastalığı Tespiti Modeli Eğitim Scripti
EfficientNet + Class Imbalance + Early Stopping + Metrikler
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Proje kök dizinini path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.disease_model_optimized import EfficientNetDiseaseModel
from src.data_loader_optimized import CancerDatasetOptimized, get_cancer_transforms_optimized, create_weighted_dataloader


class EarlyStopping:
    """Early stopping için callback"""
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Bir epoch eğitim"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # İstatistikler
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def validate(model, dataloader, criterion, device):
    """Validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def calculate_metrics(y_true, y_pred, class_names):
    """Detaylı metrikleri hesapla"""
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Per-class F1 scores
    f1_scores = {}
    for i, class_name in enumerate(class_names):
        f1_scores[class_name] = report[class_name]['f1-score']
    
    # Overall metrics
    overall_acc = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    weighted_f1 = report['weighted avg']['f1-score']
    
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'f1_scores': f1_scores,
        'accuracy': overall_acc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """Confusion matrix'i görselleştir"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def train_model(
    csv_path: str = None,
    images_dir: str = None,
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    num_classes: int = 7,
    model_name: str = 'efficientnet_b0',
    patience: int = 5
):
    """Optimize edilmiş model eğitimi"""
    print("=" * 70)
    print("OPTIMIZE EDILMIS CILT HASTALIGI TESPITI MODELI EGITIMI")
    print("=" * 70)
    
    # Varsayılan yollar
    if csv_path is None:
        csv_path = os.path.join(project_root, "data", "cancer", "GroundTruth.csv")
    if images_dir is None:
        images_dir = os.path.join(project_root, "data", "cancer", "images")
    
    # Cihaz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanilan cihaz: {device}")
    
    # Veri setleri
    print("\nVeri setleri yukleniyor...")
    train_dataset = CancerDatasetOptimized(
        csv_path=csv_path,
        images_dir=images_dir,
        transform=get_cancer_transforms_optimized(train=True),
        train=True
    )
    
    val_dataset = CancerDatasetOptimized(
        csv_path=csv_path,
        images_dir=images_dir,
        transform=get_cancer_transforms_optimized(train=False),
        train=False
    )
    
    # Class weights hesapla
    class_weights = train_dataset.get_class_weights()
    print(f"\nSinif agirliklari:")
    for i, class_name in enumerate(train_dataset.CLASS_NAMES):
        print(f"  {class_name}: {class_weights[i]:.3f}")
    
    # Weighted DataLoader
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2
    
    train_loader = create_weighted_dataloader(
        train_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    # Model
    print(f"\nModel olusturuluyor: {model_name}...")
    model = EfficientNetDiseaseModel(num_classes=num_classes, model_name=model_name)
    
    # İlk aşama: Feature extractor'ı freeze et
    print("1. Asama: Feature extractor freeze ediliyor...")
    model.freeze_backbone()
    model = model.to(device)
    
    # Weighted Loss
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer (sadece classifier için)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4
    )
    
    # Early Stopping
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    
    # İlk aşama eğitim (frozen backbone)
    print(f"\n1. Asama egitim basliyor (Feature extractor frozen)...")
    best_val_acc = 0.0
    
    for epoch in range(epochs // 2):  # İlk yarı frozen
        print(f"\nEpoch {epoch+1}/{epochs // 2}")
        print("-" * 70)
        
        train_loss, train_acc, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        print(f"\nEpoch {epoch+1} Ozeti:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping kontrolü
        if early_stopping(val_loss, model):
            print(f"Early stopping! En iyi model geri yuklendi.")
            break
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    # İkinci aşama: Fine-tuning (son katmanları unfreeze)
    print(f"\n2. Asama: Fine-tuning basliyor (Son katmanlar unfreeze)...")
    model.unfreeze_last_layers(num_layers=3)
    
    # Optimizer'ı tüm parametreler için güncelle
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate / 10,  # Daha düşük learning rate
        weight_decay=1e-4
    )
    
    # Fine-tuning eğitimi
    for epoch in range(epochs // 2, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 70)
        
        train_loss, train_acc, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        print(f"\nEpoch {epoch+1} Ozeti:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping kontrolü
        if early_stopping(val_loss, model):
            print(f"Early stopping! En iyi model geri yuklendi.")
            break
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    
    # Final değerlendirme
    print("\n" + "=" * 70)
    print("FINAL DEĞERLENDIRME")
    print("=" * 70)
    
    # Validation seti üzerinde metrikler
    val_loss, val_acc, val_preds, val_labels = validate(
        model, val_loader, criterion, device
    )
    
    metrics = calculate_metrics(val_labels, val_preds, train_dataset.CLASS_NAMES)
    
    print(f"\nGenel Metrikler:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    
    print(f"\nSinif Bazli F1 Skorlari:")
    for class_name, f1 in metrics['f1_scores'].items():
        print(f"  {class_name}: {f1:.4f}")
    
    # Özellikle MEL (Melanom) için
    mel_f1 = metrics['f1_scores'].get('Melanom', 0)
    print(f"\nMelanom (MEL) F1 Score: {mel_f1:.4f}")
    
    # Confusion matrix kaydet
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    cm_path = os.path.join(model_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics['confusion_matrix'], train_dataset.CLASS_NAMES, cm_path)
    print(f"\nConfusion matrix kaydedildi: {cm_path}")
    
    # Model kaydet
    model_path = os.path.join(model_dir, "skin_disease_model_optimized.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model kaydedildi: {model_path}")
    
    # Classification report kaydet
    report_path = os.path.join(model_dir, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(classification_report(val_labels, val_preds, target_names=train_dataset.CLASS_NAMES))
    print(f"Classification report kaydedildi: {report_path}")
    
    print("\n" + "=" * 70)
    print("EGITIM TAMAMLANDI!")
    print(f"En iyi validation accuracy: {best_val_acc:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize Edilmis Cilt Hastaligi Modeli Egitimi')
    parser.add_argument('--csv', type=str, default=None, help='CSV dosya yolu')
    parser.add_argument('--images', type=str, default=None, help='Goruntu klasoru yolu')
    parser.add_argument('--epochs', type=int, default=25, help='Epoch sayisi')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model', type=str, default='efficientnet_b0', choices=['efficientnet_b0', 'efficientnet_b1'], help='Model tipi')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    
    args = parser.parse_args()
    
    train_model(
        csv_path=args.csv,
        images_dir=args.images,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_name=args.model,
        patience=args.patience
    )

