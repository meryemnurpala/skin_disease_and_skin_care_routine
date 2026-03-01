"""
Veri yükleme modüllerini test et
"""

import os
import sys

# Proje kök dizinini path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data_loader import CancerDataset, SkinTypeDataset, get_cancer_transforms, get_skin_type_transforms

def test_cancer_dataset():
    """Cancer dataset'i test et"""
    print("=" * 60)
    print("Cancer Dataset Testi")
    print("=" * 60)
    
    csv_path = os.path.join(project_root, "data", "cancer", "GroundTruth.csv")
    images_dir = os.path.join(project_root, "data", "cancer", "images")
    
    if not os.path.exists(csv_path):
        print(f"CSV dosyası bulunamadı: {csv_path}")
        return
    
    if not os.path.exists(images_dir):
        print(f"Görüntü klasörü bulunamadı: {images_dir}")
        return
    
    print(f"CSV: {csv_path}")
    print(f"Images: {images_dir}")
    
    try:
        train_dataset = CancerDataset(
            csv_path=csv_path,
            images_dir=images_dir,
            transform=get_cancer_transforms(train=True),
            train=True
        )
        
        print(f"\nTrain seti: {len(train_dataset)} örnek")
        
        # İlk örneği göster
        if len(train_dataset) > 0:
            image, label = train_dataset[0]
            print(f"İlk örnek - Label: {label}, Image shape: {image.shape}")
            print(f"Sınıf adı: {train_dataset.CLASS_NAMES[label]}")
        
        val_dataset = CancerDataset(
            csv_path=csv_path,
            images_dir=images_dir,
            transform=get_cancer_transforms(train=False),
            train=False
        )
        
        print(f"Validation seti: {len(val_dataset)} örnek")
        print("[OK] Cancer dataset basariyla yuklendi!")
        
    except Exception as e:
        print(f"[HATA] Hata: {e}")
        import traceback
        traceback.print_exc()


def test_skin_type_dataset():
    """Skin Type dataset'i test et"""
    print("\n" + "=" * 60)
    print("Skin Type Dataset Testi")
    print("=" * 60)
    
    data_dir = os.path.join(project_root, "data", "skin_type")
    
    if not os.path.exists(data_dir):
        print(f"Veri seti klasörü bulunamadı: {data_dir}")
        return
    
    print(f"Data dir: {data_dir}")
    
    try:
        train_dataset = SkinTypeDataset(
            data_dir=data_dir,
            split='train',
            transform=get_skin_type_transforms(train=True)
        )
        
        print(f"\nTrain seti: {len(train_dataset)} örnek")
        
        # İlk örneği göster
        if len(train_dataset) > 0:
            image, label = train_dataset[0]
            print(f"İlk örnek - Label: {label}, Image shape: {image.shape}")
            print(f"Sınıf adı: {train_dataset.CLASS_NAMES[label]}")
        
        val_dataset = SkinTypeDataset(
            data_dir=data_dir,
            split='valid',
            transform=get_skin_type_transforms(train=False)
        )
        
        print(f"Validation seti: {len(val_dataset)} örnek")
        print("[OK] Skin Type dataset basariyla yuklendi!")
        
    except Exception as e:
        print(f"[HATA] Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_cancer_dataset()
    test_skin_type_dataset()
    
    print("\n" + "=" * 60)
    print("Test tamamlandı!")
    print("=" * 60)

