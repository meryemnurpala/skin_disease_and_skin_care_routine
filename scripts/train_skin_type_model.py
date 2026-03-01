"""
Cilt Tipi Analizi Modeli Eğitim Scripti
Skin Type veri seti ile model eğitimi
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm

# Proje kök dizinini path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.skin_type_model import SkinTypeCNN
from src.data_loader import SkinTypeDataset, get_skin_type_transforms

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Bir epoch eğitim"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def train_model(
    data_dir: str = None,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_classes: int = 3
):
    """Model eğitimi"""
    print("=" * 60)
    print("Cilt Tipi Analizi Modeli Eğitimi")
    print("=" * 60)
    
    # Varsayılan yol
    if data_dir is None:
        data_dir = os.path.join(project_root, "data", "skin_type")
    
    # Cihaz
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    
    # Veri setleri
    print("\nVeri setleri yükleniyor...")
    train_dataset = SkinTypeDataset(
        data_dir=data_dir,
        split='train',
        transform=get_skin_type_transforms(train=True)
    )
    
    val_dataset = SkinTypeDataset(
        data_dir=data_dir,
        split='valid',
        transform=get_skin_type_transforms(train=False)
    )
    
    # Windows uyumluluğu için num_workers=0
    import platform
    num_workers = 0 if platform.system() == 'Windows' else 2
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Model
    print("\nModel oluşturuluyor...")
    model = SkinTypeCNN(num_classes=num_classes)
    model = model.to(device)
    
    # Loss ve optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Eğitim
    print(f"\nEğitim başlıyor ({epochs} epoch)...")
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate update
        scheduler.step()
        
        print(f"\nEpoch {epoch+1} Özeti:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # En iyi modeli kaydet
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_dir = os.path.join(project_root, "models")
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "skin_type_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ En iyi model kaydedildi: {model_path} (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "=" * 60)
    print("Eğitim tamamlandı!")
    print(f"En iyi validation accuracy: {best_val_acc:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Cilt Tipi Modeli Eğitimi')
    parser.add_argument('--data-dir', type=str, default=None, help='Veri seti klasörü yolu')
    parser.add_argument('--epochs', type=int, default=30, help='Epoch sayısı')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

