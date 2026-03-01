"""
Cilt Bakımı AI Uygulaması
Streamlit Frontend + Backend Logic
"""

import streamlit as st
import sys
import os
from PIL import Image
import io

# Proje kök dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.disease_model import DiseaseModel
from src.skin_type_model import SkinTypeModel
from src.routine_engine import RoutineEngine

# Sayfa yapılandırması
st.set_page_config(
    page_title="Cilt Bakımı AI Uygulaması",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stilleri - Lila-Pembe Tema
st.markdown("""
<style>
    /* Ana arka plan gradyanı */
    .stApp {
        background: linear-gradient(135deg, #f5e6ff 0%, #ffe6f5 50%, #e6f0ff 100%);
    }
    
    /* Header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #b794f6 0%, #f687b3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 1rem;
        text-shadow: 2px 2px 4px rgba(183, 148, 246, 0.3);
    }
    
    /* Uyarı kutusu */
    .warning-box {
        background: linear-gradient(135deg, #fff0f5 0%, #ffe6f5 100%);
        border: 2px solid #f687b3;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(246, 135, 179, 0.2);
    }
    
    /* Bilgi kutusu */
    .info-box {
        background: linear-gradient(135deg, #f0e6ff 0%, #e6f0ff 100%);
        border: 2px solid #b794f6;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(183, 148, 246, 0.2);
    }
    
    /* Başarı kutusu */
    .success-box {
        background: linear-gradient(135deg, #e6ffe6 0%, #f0fff0 100%);
        border: 2px solid #a8e6cf;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(168, 230, 207, 0.2);
    }
    
    /* Risk seviyeleri */
    .risk-high {
        color: #e91e63;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(233, 30, 99, 0.3);
    }
    .risk-medium {
        color: #f687b3;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(246, 135, 179, 0.3);
    }
    .risk-low {
        color: #9c88ff;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(156, 136, 255, 0.3);
    }
    
    /* Butonlar */
    .stButton > button {
        background: linear-gradient(135deg, #b794f6 0%, #f687b3 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(183, 148, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(183, 148, 246, 0.4);
    }
    
    /* Sekmeler */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f0e6ff 0%, #ffe6f5 100%);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #b794f6 0%, #f687b3 100%);
        color: white;
    }
    
    /* File uploader */
    .uploadedFile {
        border: 2px dashed #b794f6;
        border-radius: 15px;
        padding: 20px;
        background: rgba(255, 255, 255, 0.7);
    }
    
    /* Metrikler */
    .metric-container {
        background: linear-gradient(135deg, #f0e6ff 0%, #ffe6f5 100%);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(183, 148, 246, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Model yükleme (cache ile)
@st.cache_resource
def load_models():
    """Modelleri yükle ve cache'le"""
    # Önce optimize edilmiş modeli dene, yoksa eski modeli kullan
    optimized_path = "models/skin_disease_model_optimized.pth"
    default_path = "models/skin_disease_model.pth"
    
    if os.path.exists(optimized_path):
        from src.disease_model_optimized import OptimizedDiseaseModel
        disease_model = OptimizedDiseaseModel(model_path=optimized_path)
    else:
        disease_model = DiseaseModel(model_path=default_path)
    
    skin_type_model = SkinTypeModel(model_path="models/skin_type_model.pth")
    routine_engine = RoutineEngine()
    return disease_model, skin_type_model, routine_engine

# Modelleri yükle
try:
    disease_model, skin_type_model, routine_engine = load_models()
except Exception as e:
    st.error(f"Model yükleme hatası: {e}")
    st.stop()

# Ana başlık
st.markdown('<div class="main-header">✨ Cilt Bakımı AI Uygulaması</div>', unsafe_allow_html=True)

# Etik uyarı
st.markdown("""
<div class="warning-box">
    <h4>⚠️ ÖNEMLİ UYARI</h4>
    <p><strong>Bu uygulama tıbbi tanı koymaz, sadece bilgilendirme amaçlıdır.</strong></p>
    <p>Şüpheli lezyonlar veya cilt sorunları için mutlaka bir dermatoloğa başvurun.</p>
</div>
""", unsafe_allow_html=True)

# Sekme sistemi
tab1, tab2 = st.tabs(["🔍 Cilt Hastalığı Tespiti", "💆 Cilt Tipi Analizi ve Bakım Rutini"])

# ==================== TAB 1: CİLT HASTALIĞI TESPİTİ ====================
with tab1:
    st.header("🔍 Cilt Hastalığı Tespiti")
    st.markdown("""
    <div class="info-box">
        <p>Cilt lezyonu, ben, yara veya akne fotoğrafınızı yükleyin. 
        AI modeli görüntüyü analiz ederek olası hastalık sınıfını ve risk durumunu değerlendirecektir.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Görüntü yükleyin (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Cilt lezyonu fotoğrafınızı seçin"
    )
    
    if uploaded_file is not None:
        # Görüntüyü göster
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Yüklenen Görüntü", use_container_width=True)
        
        with col2:
            if st.button("🔬 Analiz Et", type="primary", use_container_width=True):
                with st.spinner("Görüntü analiz ediliyor..."):
                    result = disease_model.predict(image)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("Analiz tamamlandı!")
                        
                        # Sonuçları göster
                        st.markdown("### 📊 Analiz Sonuçları")
                        
                        # Hastalık sınıfı
                        st.markdown(f"**Hastalık Sınıfı:** {result['disease_class']}")
                        
                        # Risk seviyesi
                        risk_color = "risk-low"
                        if result['risk_level'] == "Çok Yüksek":
                            risk_color = "risk-high"
                        elif result['risk_level'] == "Yüksek":
                            risk_color = "risk-high"
                        elif result['risk_level'] == "Düşük-Orta":
                            risk_color = "risk-medium"
                        
                        st.markdown(f"**Risk Durumu:** <span class='{risk_color}'>{result['risk_level']}</span>", 
                                  unsafe_allow_html=True)
                        st.markdown(f"**Güven Skoru:** {result['confidence']}%")
                        
                        # Belirsizlik uyarısı
                        if result.get('is_uncertain', False):
                            st.warning(result.get('warning', 'Model tahmininde emin degil. Lutfen bir dermatologa basvurun.'))
                        
                        # Açıklama
                        st.markdown("### 📝 Açıklama")
                        st.info(result['description'])
                        
                        # Tüm olasılıklar
                        with st.expander("📈 Detaylı Olasılık Dağılımı"):
                            for disease, prob in result['all_probabilities'].items():
                                st.progress(prob / 100, text=f"{disease}: {prob}%")
                        
                        # Uyarı
                        st.markdown("""
                        <div class="warning-box">
                            <p><strong>⚠️ Bu uygulama tıbbi tanı koymaz. Şüpheli lezyonlar için dermatoloğa başvurun.</strong></p>
                        </div>
                        """, unsafe_allow_html=True)

# ==================== TAB 2: CİLT TİPİ ANALİZİ VE BAKIM RUTİNİ ====================
with tab2:
    st.header("💆 Cilt Tipi Analizi ve Kişiselleştirilmiş Bakım Rutini")
    
    # Aşama 1: Fotoğraf Yükleme
    st.markdown("### 📸 Aşama 1: Fotoğraf Yükleme")
    st.markdown("Yüz/cilt fotoğrafınızı yükleyin. Model cilt tipinizi analiz edecektir.")
    
    skin_image = st.file_uploader(
        "Cilt fotoğrafı yükleyin",
        type=['jpg', 'jpeg', 'png'],
        key="skin_type_image",
        help="Yüz veya cilt bölgesi fotoğrafınızı seçin"
    )
    
    skin_type_result = None
    
    if skin_image is not None:
        image_skin = Image.open(skin_image)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image_skin, caption="Yüklenen Cilt Fotoğrafı", use_container_width=True)
        
        with col2:
            if st.button("🔍 Cilt Tipini Analiz Et", type="primary", key="analyze_skin"):
                with st.spinner("Cilt tipi analiz ediliyor..."):
                    skin_type_result = skin_type_model.predict(image_skin)
                    
                    if "error" in skin_type_result:
                        st.error(skin_type_result["error"])
                    else:
                        st.success("Analiz tamamlandı!")
                        st.markdown(f"**Tespit Edilen Cilt Tipi:** {skin_type_result['skin_type']}")
                        st.markdown(f"**Güven Skoru:** {skin_type_result['confidence']}%")
                        st.info(skin_type_result['description'])
    
    # Aşama 2: Ek Soru Formu
    st.markdown("---")
    st.markdown("### 📋 Aşama 2: Ek Bilgiler")
    st.markdown("Daha kişiselleştirilmiş öneriler için lütfen aşağıdaki soruları yanıtlayın.")
    
    # Cilt sorunları (çoklu seçim)
    st.markdown("**Genelde en rahatsız olduğunuz cilt problemi nedir? (Birden fazla seçebilirsiniz)**")
    concerns_options = [
        "kuruluk", "akne", "siyah nokta", "komedon", "kızarıklık",
        "hassasiyet", "güneş lekesi", "mat görünüm", "T bölgesi yağlanması",
        "geniş gözenek", "ince çizgiler", "hiperpigmentasyon", "pullanma",
        "dermatitis", "egzama eğilimi"
    ]
    
    selected_concerns = st.multiselect(
        "Cilt sorunlarınızı seçin:",
        concerns_options,
        default=[],
        help="Size uygun olanları seçin"
    )
    
    # Ek sorular
    col1, col2 = st.columns(2)
    
    with col1:
        barrier_damaged = st.radio(
            "Cilt bariyerinizin zarar gördüğünü düşünüyor musunuz?",
            ["Evet", "Hayır", "Emin değilim"],
            index=2
        )
        
        uses_sunscreen = st.radio(
            "Güneş kremi kullanır mısınız?",
            ["Evet, düzenli", "Ara sıra", "Hayır"],
            index=1
        )
        
        water_intake = st.selectbox(
            "Günde kaç litre su içersiniz?",
            ["Az (1 litre altı)", "Orta (1-2 litre)", "İyi (2+ litre)"],
            index=1
        )
    
    with col2:
        diet = st.selectbox(
            "Beslenme alışkanlıklarınız nasıl?",
            ["İyi (dengeli, sağlıklı)", "Orta (karışık)", "Kötü (fast food, şekerli)"],
            index=1
        )
        
        environmental_sensitivity = st.radio(
            "Çevresel faktörlere (soğuk, güneş, rüzgar) hassas mısınız?",
            ["Evet", "Hayır", "Bazen"],
            index=1
        )
    
    # Aşama 3: Rutin Önerisi
    st.markdown("---")
    st.markdown("### ✨ Aşama 3: Kişiselleştirilmiş Bakım Rutini")
    
    if st.button("🎯 Bakım Rutinimi Oluştur", type="primary", use_container_width=True):
        if not skin_type_result or "error" in skin_type_result:
            st.warning("⚠️ Lütfen önce cilt tipi analizini tamamlayın!")
        else:
            # Form verilerini işle
            barrier_damaged_bool = barrier_damaged == "Evet"
            uses_sunscreen_bool = uses_sunscreen == "Evet, düzenli"
            
            water_map = {
                "Az (1 litre altı)": "az",
                "Orta (1-2 litre)": "orta",
                "İyi (2+ litre)": "iyi"
            }
            water_intake_processed = water_map.get(water_intake, "orta")
            
            diet_map = {
                "İyi (dengeli, sağlıklı)": "iyi",
                "Orta (karışık)": "orta",
                "Kötü (fast food, şekerli)": "kötü"
            }
            diet_processed = diet_map.get(diet, "orta")
            
            env_sensitive = environmental_sensitivity in ["Evet", "Bazen"]
            
            # Rutin oluştur
            with st.spinner("Kişiselleştirilmiş rutin oluşturuluyor..."):
                routine = routine_engine.generate_routine(
                    skin_type=skin_type_result['skin_type'],
                    concerns=selected_concerns,
                    barrier_damaged=barrier_damaged_bool,
                    uses_sunscreen=uses_sunscreen_bool,
                    water_intake=water_intake_processed,
                    diet=diet_processed,
                    environmental_sensitivity=env_sensitive
                )
                
                # Rutini göster
                st.success("✅ Kişiselleştirilmiş bakım rutininiz hazır!")
                
                # Rutin detayları
                st.markdown("## 🌅 SABAH RUTİNİ")
                for i, step in enumerate(routine["morning"], 1):
                    st.markdown(f"{i}. {step}")
                
                st.markdown("## 🌙 AKŞAM RUTİNİ")
                for i, step in enumerate(routine["evening"], 1):
                    st.markdown(f"{i}. {step}")
                
                if routine["weekly"]:
                    st.markdown("## 📅 HAFTALIK DESTEK")
                    for i, step in enumerate(routine["weekly"], 1):
                        st.markdown(f"{i}. {step}")
                
                if routine["ingredients_to_avoid"]:
                    st.markdown("## ⛔ KAÇINILMASI GEREKEN İÇERİKLER")
                    for item in routine["ingredients_to_avoid"]:
                        st.markdown(f"- {item}")
                
                if routine["warnings"]:
                    st.markdown("## ⚠️ UYARILAR")
                    for warning in routine["warnings"]:
                        st.markdown(f"- {warning}")
                
                if routine["general_tips"]:
                    st.markdown("## 💡 GENEL İPUÇLARI")
                    for tip in routine["general_tips"]:
                        st.markdown(f"- {tip}")
                
                # Genel uyarı
                st.markdown("""
                <div class="warning-box">
                    <p><strong>⚠️ Güçlü retinoid ve asit kombinasyonlarını dermatoloğa danışmadan kullanmayın.</strong></p>
                </div>
                """, unsafe_allow_html=True)
                
                # İndirme butonu (opsiyonel)
                routine_text = routine_engine.format_routine_output(routine)
                st.download_button(
                    label="📥 Rutini İndir (TXT)",
                    data=routine_text,
                    file_name="cilt_bakim_rutini.txt",
                    mime="text/plain"
                )

# Sidebar bilgileri
with st.sidebar:
    st.markdown("## 📖 Hakkında")
    st.markdown("""
    Bu uygulama AI destekli cilt analizi ve kişiselleştirilmiş bakım rutini önerisi sunar.
    
    ### Özellikler:
    - 🔍 Cilt hastalığı tespiti
    - 💆 Cilt tipi analizi
    - ✨ Kişiselleştirilmiş bakım rutini
    
    ### ⚠️ Önemli:
    Bu uygulama tıbbi tanı koymaz. 
    Şüpheli durumlar için mutlaka dermatoloğa başvurun.
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 Teknik Detaylar")
    st.markdown("""
    - **Backend:** Python (PyTorch)
    - **Frontend:** Streamlit
    - **Modeller:** CNN (Convolutional Neural Network)
    """)

