# ✨ Cilt Bakımı AI Uygulaması

AI destekli cilt hastalığı tespiti ve kişiselleştirilmiş cilt bakım rutini önerisi sunan web uygulaması.

## 🎯 Özellikler

### 1️⃣ Cilt Hastalığı Tespiti Modülü
- Cilt lezyonu, ben, yara veya akne görüntüsü analizi
- PyTorch tabanlı CNN modeli ile hastalık sınıflandırması
- Risk seviyesi değerlendirmesi
- Detaylı açıklama ve olasılık dağılımı

### 2️⃣ Cilt Tipi Analizi ve Bakım Rutini Modülü
- Cilt tipi sınıflandırması (Kuru, Normal, Yağlı)
- Kapsamlı soru formu ile kişiselleştirme
- Kural tabanlı uzman sistem ile rutin önerisi
- Sabah, akşam ve haftalık bakım rutinleri
- Kaçınılması gereken içerikler ve uyarılar

## 🏗️ Proje Yapısı

```
skin_care_ai/
├── app.py                 # Ana Streamlit uygulaması
├── requirements.txt       # Python bağımlılıkları
├── README.md             # Bu dosya
├── models/               # Model dosyaları
│   ├── skin_disease_model.pth
│   └── skin_type_model.pth
├── src/                  # Kaynak kodlar
│   ├── disease_model.py      # Hastalık tespiti modeli
│   ├── skin_type_model.py   # Cilt tipi analizi modeli
│   └── routine_engine.py    # Rutin öneri motoru
└── data/                 # Veri setleri (opsiyonel)
```

## 🚀 Hızlı Başlangıç

### Adım 1: Bağımlılıkları Yükle
```bash
pip install -r requirements.txt
```

### Adım 2: Veri Setlerini Test Et
```bash
python scripts/test_data_loader.py
```

### Adım 3: Modelleri Eğit (İlk Kurulum)
```bash
# Hastalık tespiti modeli
python scripts/train_disease_model.py --epochs 20 --batch-size 32

# Cilt tipi analizi modeli
python scripts/train_skin_type_model.py --epochs 30 --batch-size 32
```

### Adım 4: Web Uygulamasını Başlat
```bash
streamlit run app.py
```

**📖 Detaylı kurulum kılavuzu için:** [KURULUM_KILAVUZU.md](KURULUM_KILAVUZU.md) dosyasına bakın.

## 🚀 Kurulum (Detaylı)

### 1. Gereksinimler

- Python 3.8 veya üzeri
- pip (Python paket yöneticisi)

### 2. Bağımlılıkları Yükleme

```bash
pip install -r requirements.txt
```

### 3. Model Dosyalarını Hazırlama

Model dosyaları (`skin_disease_model.pth` ve `skin_type_model.pth`) `models/` klasörüne yerleştirilmelidir.

**Model Eğitimi:**

Veri setleri `data/` klasöründe mevcuttur. Modelleri eğitmek için:

```bash
# Hastalık tespiti modeli için (Cancer dataset)
python scripts/train_disease_model.py --epochs 20 --batch-size 32

# Cilt tipi analizi modeli için (Skin Type dataset)
python scripts/train_skin_type_model.py --epochs 30 --batch-size 32
```

**Eğitim Parametreleri:**
- `--epochs`: Epoch sayısı (varsayılan: 20/30)
- `--batch-size`: Batch size (varsayılan: 32)
- `--lr`: Learning rate (varsayılan: 0.001)
- `--csv`: CSV dosya yolu (hastalık modeli için)
- `--images`: Görüntü klasörü yolu (hastalık modeli için)
- `--data-dir`: Veri seti klasörü yolu (cilt tipi modeli için)

**Not:** Eğer model dosyalarınız yoksa, uygulama varsayılan ağırlıklarla çalışacaktır ancak performans düşük olacaktır.

### 4. Uygulamayı Çalıştırma

```bash
# Yöntem 1: Streamlit ile direkt
streamlit run app.py

# Yöntem 2: Başlatıcı script ile
python run.py
```

Tarayıcınızda otomatik olarak açılacaktır (genellikle `http://localhost:8501`).

## 📋 Kullanım

### Model Eğitimi (İlk Kurulum)

İlk kullanımda modelleri eğitmeniz gerekmektedir:

```bash
# Veri yükleme modüllerini test et
python scripts/test_data_loader.py

# Hastalık tespiti modelini eğit (yaklaşık 20-30 dakika)
python scripts/train_disease_model.py --epochs 20 --batch-size 32

# Cilt tipi analizi modelini eğit (yaklaşık 15-20 dakika)
python scripts/train_skin_type_model.py --epochs 30 --batch-size 32
```

**Not:** GPU varsa eğitim çok daha hızlı olacaktır. CPU ile eğitim uzun sürebilir.

### Web Uygulaması Kullanımı

#### Cilt Hastalığı Tespiti

1. "Cilt Hastalığı Tespiti" sekmesine gidin
2. Cilt lezyonu fotoğrafınızı yükleyin
3. "Analiz Et" butonuna tıklayın
4. Sonuçları inceleyin

#### Cilt Tipi Analizi ve Bakım Rutini

1. "Cilt Tipi Analizi ve Bakım Rutini" sekmesine gidin
2. Cilt fotoğrafınızı yükleyin ve analiz edin
3. Soru formunu doldurun
4. "Bakım Rutinimi Oluştur" butonuna tıklayın
5. Kişiselleştirilmiş rutininizi görüntüleyin ve indirin

## ⚠️ ÖNEMLİ UYARILAR

- **Bu uygulama tıbbi tanı koymaz, sadece bilgilendirme amaçlıdır.**
- Şüpheli lezyonlar veya cilt sorunları için mutlaka bir dermatoloğa başvurun.
- Güçlü retinoid ve asit kombinasyonlarını dermatoloğa danışmadan kullanmayın.
- Uygulama sonuçları profesyonel tıbbi görüşün yerini tutmaz.

## 🔧 Teknik Detaylar

### Backend
- **Framework:** Streamlit (Python)
- **AI Framework:** PyTorch
- **Görüntü İşleme:** PIL (Pillow), torchvision

### Modeller
- **Hastalık Tespiti:** CNN (7 sınıf)
  - Aktinik Keratoz (AKIEC)
  - Bazal Hücreli Karsinom (BCC)
  - Benign Keratoz (BKL)
  - Dermatofibrom (DF)
  - Melanom (MEL)
  - Melanositik Nevüs (NV)
  - Vasküler Lezyon (VASC)
- **Cilt Tipi:** CNN (3 sınıf: Kuru, Normal, Yağlı)

### Veri Setleri
- **Cancer Dataset:** HAM10000 (10,015 görüntü, 7 sınıf)
- **Skin Type Dataset:** Train/Valid/Test split (2,756 train, 262 valid, 134 test)

### Rutin Öneri Sistemi
- Kural tabanlı uzman sistem
- Cilt tipi ve sorunlara göre kişiselleştirme
- İçerik önerileri ve uyarılar

## 📝 Lisans

Bu proje eğitim ve bilgilendirme amaçlıdır. Tıbbi kullanım için uygun değildir.

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen önce bir issue açın veya pull request gönderin.

## 📧 İletişim

Sorularınız için meryemnur6969@gmail.com

---

**Not:** Bu uygulama tıbbi tanı koymaz. Her zaman profesyonel tıbbi görüş alın.

