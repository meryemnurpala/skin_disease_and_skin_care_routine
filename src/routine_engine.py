"""
Kişiselleştirilmiş Cilt Bakım Rutini Öneri Motoru
Kural tabanlı uzman sistem
"""

from typing import Dict, List, Set
from enum import Enum


class SkinType(Enum):
    DRY = "Kuru"
    NORMAL = "Normal"
    OILY = "Yağlı"


class RoutineEngine:
    """Cilt bakım rutini öneri motoru"""
    
    # İçerik kütüphanesi
    INGREDIENTS = {
        "salisilik_asit": {
            "name": "Salisilik Asit (BHA)",
            "benefits": ["Gözenek temizliği", "Siyah nokta azaltma", "Akne kontrolü"],
            "strength": "Orta",
            "warnings": ["Güneş hassasiyeti artırabilir", "Kuru ciltlerde dikkatli kullanılmalı"]
        },
        "niacinamide": {
            "name": "Niacinamide",
            "benefits": ["Gözenek sıkılaştırma", "Yağ kontrolü", "Cilt bariyeri güçlendirme"],
            "strength": "Hafif",
            "warnings": []
        },
        "hyaluronik_asit": {
            "name": "Hyaluronik Asit",
            "benefits": ["Yoğun nemlendirme", "Cilt dolgunluğu", "Kırışıklık azaltma"],
            "strength": "Hafif",
            "warnings": []
        },
        "ceramide": {
            "name": "Ceramide",
            "benefits": ["Cilt bariyeri onarımı", "Nem tutma", "Hassasiyet azaltma"],
            "strength": "Hafif",
            "warnings": []
        },
        "retinoid": {
            "name": "Retinoid (A Vitamini)",
            "benefits": ["Kırışıklık azaltma", "Akne tedavisi", "Cilt yenileme"],
            "strength": "Güçlü",
            "warnings": ["Dermatoloğa danışmadan kullanmayın", "Güneş hassasiyeti", "Hamilelikte kullanılmamalı"]
        },
        "vitamin_c": {
            "name": "Vitamin C",
            "benefits": ["Antioksidan", "Leke açma", "Kollajen üretimi"],
            "strength": "Orta",
            "warnings": ["Güneş koruması ile kullanılmalı"]
        },
        "azelaik_asit": {
            "name": "Azelaik Asit",
            "benefits": ["Akne tedavisi", "Anti-inflamatuar", "Leke açma"],
            "strength": "Orta",
            "warnings": ["Hassas ciltlerde tahriş yapabilir"]
        },
        "peptid": {
            "name": "Peptid",
            "benefits": ["Kollajen üretimi", "Cilt sıkılaştırma", "Nem tutma"],
            "strength": "Hafif",
            "warnings": []
        },
        "aha": {
            "name": "AHA (Glikolik/Laktik Asit)",
            "benefits": ["Eksfoliasyon", "Cilt yenileme", "Mat görünüm azaltma"],
            "strength": "Orta",
            "warnings": ["Güneş hassasiyeti", "Hassas ciltlerde dikkatli kullanılmalı"]
        }
    }
    
    def __init__(self):
        pass
    
    def generate_routine(self, 
                        skin_type: str,
                        concerns: List[str],
                        barrier_damaged: bool = False,
                        uses_sunscreen: bool = False,
                        water_intake: str = "orta",
                        diet: str = "orta",
                        environmental_sensitivity: bool = False) -> Dict:
        """Kişiselleştirilmiş bakım rutini oluştur"""
        
        routine = {
            "morning": [],
            "evening": [],
            "weekly": [],
            "ingredients_to_avoid": [],
            "warnings": [],
            "general_tips": []
        }
        
        # Cilt tipine göre temel öneriler
        if skin_type == "Kuru":
            routine["morning"].extend([
                "Nazik temizleyici (köpürmeyen, sülfatsız)",
                "Hyaluronik asit serumu",
                "Ceramide içeren nemlendirici",
                "SPF 30+ güneş kremi (nemlendirici bazlı)"
            ])
            routine["evening"].extend([
                "Yağ bazlı temizleyici veya misel suyu",
                "Nazik temizleyici",
                "Hyaluronik asit + Peptid serumu",
                "Yoğun nemlendirici (ceramide, shea butter içeren)",
                "Göz çevresi kremi"
            ])
            routine["weekly"].append("Haftada 1-2 kez nazik eksfoliasyon (laktik asit)")
            
        elif skin_type == "Yağlı":
            routine["morning"].extend([
                "Jel bazlı temizleyici",
                "Niacinamide serumu",
                "Yağsız nemlendirici (gel veya su bazlı)",
                "Matlaştırıcı SPF 30+ (non-comedogenic)"
            ])
            routine["evening"].extend([
                "Çift temizleme (yağ + jel)",
                "Salisilik asit (BHA) serumu veya tonik",
                "Niacinamide serumu",
                "Yağsız nemlendirici",
                "Gözenek kontrolü için retinol (düşük konsantrasyon)"
            ])
            routine["weekly"].extend([
                "Haftada 2-3 kez salisilik asit maskesi",
                "Haftada 1 kez kil maskesi (gözenek temizliği)"
            ])
            
        else:  # Normal
            routine["morning"].extend([
                "Nazik temizleyici",
                "Antioksidan serumu (Vitamin C)",
                "Dengeli nemlendirici",
                "SPF 30+ güneş kremi"
            ])
            routine["evening"].extend([
                "Temizleyici",
                "Niacinamide veya Peptid serumu",
                "Nemlendirici",
                "Göz çevresi kremi"
            ])
            routine["weekly"].append("Haftada 1-2 kez hafif eksfoliasyon")
        
        # Sorunlara göre ek öneriler
        if "akne" in concerns:
            routine["evening"].append("Azelaik asit veya Salisilik asit (akne odaklı)")
            routine["weekly"].append("Haftada 2-3 kez salisilik asit maskesi")
            routine["ingredients_to_avoid"].extend(["Ağır yağlar", "Komedojenik içerikler"])
            
        if "siyah nokta" in concerns or "komedon" in concerns:
            routine["evening"].append("Salisilik asit (BHA) - gözenek temizliği için")
            routine["weekly"].append("Haftada 2 kez kil maskesi")
            
        if "kızarıklık" in concerns or "hassasiyet" in concerns:
            routine["morning"].append("Centella asiatica veya niacinamide (anti-inflamatuar)")
            routine["evening"].append("Ceramide serumu (bariyer onarımı)")
            routine["ingredients_to_avoid"].extend(["Parfüm", "Alkol", "Güçlü asitler"])
            routine["warnings"].append("Hassas cilt için ürünleri yavaş yavaş ekleyin")
            
        if "güneş lekesi" in concerns or "hiperpigmentasyon" in concerns:
            routine["morning"].append("Vitamin C serumu")
            routine["evening"].append("Azelaik asit veya Niacinamide")
            routine["warnings"].append("Mutlaka SPF kullanın ve güneşten korunun")
            
        if "mat görünüm" in concerns:
            routine["weekly"].append("Haftada 2 kez AHA eksfoliasyonu")
            routine["evening"].append("Hyaluronik asit (nem desteği)")
            
        if "T bölgesi yağlanması" in concerns:
            routine["morning"].append("T bölgesi için matlaştırıcı primer")
            routine["evening"].append("Niacinamide (yağ kontrolü)")
            
        if "geniş gözenek" in concerns:
            routine["evening"].append("Niacinamide serumu")
            routine["weekly"].append("Haftada 2 kez salisilik asit")
            
        if "ince çizgiler" in concerns:
            routine["evening"].append("Retinol (düşük konsantrasyon) veya Peptid serumu")
            routine["warnings"].append("Retinol kullanıyorsanız mutlaka SPF kullanın")
            
        if "pullanma" in concerns:
            routine["evening"].append("Yoğun nemlendirici (ceramide içeren)")
            routine["weekly"].append("Nazik eksfoliasyon (laktik asit)")
            
        if "dermatitis" in concerns or "egzama eğilimi" in concerns:
            routine["morning"].extend([
                "Çok nazik temizleyici",
                "Ceramide içeren nemlendirici"
            ])
            routine["evening"].append("Bariyer onarım kremi")
            routine["ingredients_to_avoid"].extend([
                "Parfüm", "Alkol", "Sert eksfoliyanlar", 
                "Retinoid (dermatoloğa danışmadan)"
            ])
            routine["warnings"].append("Dermatoloğa danışın")
            
        # Cilt bariyeri hasarı
        if barrier_damaged:
            routine["morning"].append("Ceramide serumu")
            routine["evening"].append("Bariyer onarım kremi (ceramide, kolesterol, yağ asitleri)")
            routine["ingredients_to_avoid"].extend(["Güçlü asitler", "Retinoid", "Sert eksfoliyanlar"])
            routine["warnings"].append("Bariyer onarımına odaklanın, aktif içerikleri geçici olarak azaltın")
            
        # Güneş kremi kullanımı
        if not uses_sunscreen:
            routine["warnings"].append("⚠️ Mutlaka günlük SPF 30+ kullanmaya başlayın!")
            routine["morning"].append("SPF 30+ güneş kremi (ZORUNLU)")
            
        # Su tüketimi
        if water_intake == "az":
            routine["general_tips"].append("Günde en az 2-2.5 litre su için")
        elif water_intake == "iyi":
            routine["general_tips"].append("Su tüketiminiz iyi, devam edin!")
            
        # Beslenme
        if diet == "kötü":
            routine["general_tips"].extend([
                "Şekerli ve işlenmiş gıdaları azaltın",
                "Omega-3 ve antioksidan içeren besinler tüketin",
                "Yeterli protein alın"
            ])
        elif diet == "iyi":
            routine["general_tips"].append("Beslenme alışkanlıklarınız cilt sağlığınızı destekliyor")
            
        # Çevresel hassasiyet
        if environmental_sensitivity:
            routine["morning"].append("Antioksidan serumu (çevresel hasara karşı)")
            routine["general_tips"].append("Soğuk ve rüzgarlı havalarda ekstra koruma kullanın")
            
        # Genel uyarılar
        routine["warnings"].append(
            "⚠️ Güçlü retinoid ve asit kombinasyonlarını dermatoloğa danışmadan kullanmayın."
        )
        
        # Rutin temizleme (tekrarları kaldır)
        routine["morning"] = list(dict.fromkeys(routine["morning"]))
        routine["evening"] = list(dict.fromkeys(routine["evening"]))
        routine["weekly"] = list(dict.fromkeys(routine["weekly"]))
        routine["ingredients_to_avoid"] = list(dict.fromkeys(routine["ingredients_to_avoid"]))
        
        return routine
    
    def format_routine_output(self, routine: Dict) -> str:
        """Rutini formatlanmış string olarak döndür"""
        output = []
        
        output.append("## 🌅 SABAH RUTİNİ\n")
        for i, step in enumerate(routine["morning"], 1):
            output.append(f"{i}. {step}")
        output.append("")
        
        output.append("## 🌙 AKŞAM RUTİNİ\n")
        for i, step in enumerate(routine["evening"], 1):
            output.append(f"{i}. {step}")
        output.append("")
        
        if routine["weekly"]:
            output.append("## 📅 HAFTALIK DESTEK\n")
            for i, step in enumerate(routine["weekly"], 1):
                output.append(f"{i}. {step}")
            output.append("")
        
        if routine["ingredients_to_avoid"]:
            output.append("## ⛔ KAÇINILMASI GEREKEN İÇERİKLER\n")
            for item in routine["ingredients_to_avoid"]:
                output.append(f"- {item}")
            output.append("")
        
        if routine["warnings"]:
            output.append("## ⚠️ UYARILAR\n")
            for warning in routine["warnings"]:
                output.append(f"- {warning}")
            output.append("")
        
        if routine["general_tips"]:
            output.append("## 💡 GENEL İPUÇLARI\n")
            for tip in routine["general_tips"]:
                output.append(f"- {tip}")
        
        return "\n".join(output)

