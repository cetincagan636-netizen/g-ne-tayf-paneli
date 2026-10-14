import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import voigt_profile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib.request
import json
from datetime import datetime, timezone

# -------------------------------------------------------------------------
# 1. SAYFA YAPILANDIRMASI
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Güneş Spektroskopisi & Canlı Uzay Havası Paneli",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------
# 2. HUBBLE UZAY TELESKOBU ARKA PLAN SEÇENEKLERİ & MODERN CAM (GLASSMORPHISM) CSS
# -------------------------------------------------------------------------
HUBBLE_BACKGROUNDS = {
    "Hubble: Karina Bulutsusu (Carina Nebula)": "https://images.unsplash.com/photo-1462331940025-496dfbfc7564?q=80&w=2048&auto=format&fit=crop",
    "Hubble: Yaratılış Sütunları & Derin Uzay": "https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2048&auto=format&fit=crop",
    "Hubble: Orion & Kozmik Toz Bulutu": "https://images.unsplash.com/photo-1506703719100-a0f3a48c0f86?q=80&w=2048&auto=format&fit=crop",
    "Hubble: Sombrero & Galaktik Alan": "https://images.unsplash.com/photo-1543722530-d2c3201371e7?q=80&w=2048&auto=format&fit=crop"
}

st.sidebar.markdown("### 🌌 Görsel Atmosfer")
secilen_arkaplan = st.sidebar.selectbox(
    "Arka Plan (Hubble Derin Uzay):",
    list(HUBBLE_BACKGROUNDS.keys()),
    index=0
)
bg_url = HUBBLE_BACKGROUNDS[secilen_arkaplan]

st.markdown(
    f"""
    <style>
    /* Hubble Arka Planı (Karartılmış Kozmik Katman ile Okunabilirlik) */
    .stApp {{
        background: linear-gradient(rgba(6, 9, 19, 0.80), rgba(10, 14, 28, 0.86)), 
                    url("{bg_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #f0f6fc;
    }}
    
    /* Yan Menü (Sidebar) Buzlu Cam Görünümü */
    section[data-testid="stSidebar"] {{
        background: rgba(11, 15, 29, 0.82) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.12);
    }}
    
    /* Üst Metrik Kartları için Glassmorphism */
    div[data-testid="stMetric"] {{
        background: rgba(18, 25, 46, 0.68);
        backdrop-filter: blur(10px);
        padding: 15px 18px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
    }}
    
    /* Bilgi Kutuları */
    div.stAlert {{
        background: rgba(15, 23, 42, 0.78) !important;
        backdrop-filter: blur(8px);
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        border-radius: 10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------------
# 3. ATOMİK VE FİZİKSEL VERİ TABANI (GÜNEŞ FRAUNHOFER ÇİZGİLERİ)
# -------------------------------------------------------------------------
SPECTRAL_LINES = {
    "Hidrojen H-Alfa (Hα - 6562.8 Å)": {
        "element": "Hidrojen (H I)",
        "lambda0": 6562.8,
        "sigma": 0.38,
        "gamma": 0.45,
        "depth": 0.82,
        "g_eff": 1.05,
        "katman": "Orta / Üst Kromosfer",
        "gecis": "Balmer n=3 → n=2",
        "renk": "#ff4b4b",
        "flare_sens": 1.35,
        "sdo_default": "AIA 304 Å (Kromosfer & Prominens - 50,000 K)",
        "aciklama": "Güneş fiziğinin en temel kromosferik çizgisidir. Güneş patlamaları (Flare) sırasında üst atmosferdeki ani ısınma ve elektron bombardımanı nedeniyle çizginin en dip noktası (çekirdeği) soğurulmadan salma (emisyon) durumuna geçer ve çukur yukarı doğru dolar."
    },
    "Kalsiyum II K (Ca II K - 3933.7 Å)": {
        "element": "İyonize Kalsiyum (Ca II)",
        "lambda0": 3933.7,
        "sigma": 0.55,
        "gamma": 0.85,
        "depth": 0.92,
        "g_eff": 1.17,
        "katman": "Üst Fotosfer & Kromosfer",
        "gecis": "Rezonans 4s ²S → 4p ²P",
        "renk": "#a855f7",
        "flare_sens": 1.60,
        "sdo_default": "AIA 1700 Å (Üst Fotosfer & Minimum Sıcaklık - 4,500 K)",
        "aciklama": "Çok geniş 'kanatlara' (Stark/basınç genişlemesi) sahip derin bir ultraviyole sınır çizgisidir. Kromosferik manyetik ağ (plage) bölgelerine ve patlamalara karşı en duyarlı çizgidir; patlama anında çekirdeğinde çok güçlü çift tepeli emisyon (K2v/K2r) oluşur."
    },
    "Sodyum I D2 (Na I D2 - 5889.9 Å)": {
        "element": "Nötr Sodyum (Na I)",
        "lambda0": 5889.9,
        "sigma": 0.12,
        "gamma": 0.15,
        "depth": 0.84,
        "g_eff": 1.33,
        "katman": "Üst Fotosfer & Alt Kromosfer",
        "gecis": "Rezonans Dubleti (3s ²S → 3p ²P)",
        "renk": "#f59e0b",
        "flare_sens": 0.65,
        "sdo_default": "AIA 1600 Å (Üst Fotosfer & Geçiş Bölgesi - 10,000 K)",
        "aciklama": "Fotosfer ile kromosfer arasındaki geçiş sınırını (minimum sıcaklık bölgesi) inceler. Büyük patlamalarda çekirdeği sığlaşır, güneş lekelerinde ise orta düzeyde manyetik genişleme gösterir."
    },
    "Magnezyum I b1 (Mg I b1 - 5183.6 Å)": {
        "element": "Nötr Magnezyum (Mg I)",
        "lambda0": 5183.6,
        "sigma": 0.14,
        "gamma": 0.22,
        "depth": 0.80,
        "g_eff": 1.25,
        "katman": "Üst Fotosfer / Alt Kromosfer",
        "gecis": "Triplet (3p ³P → 4s ³S)",
        "renk": "#10b981",
        "flare_sens": 0.55,
        "sdo_default": "AIA 1600 Å (Üst Fotosfer & Geçiş Bölgesi - 10,000 K)",
        "aciklama": "Yeşil bölgedeki ünlü Mg b üçlüsünün en güçlü üyesidir. Basınç genişlemesi (kanatlar) fotosferik yoğunluğu, merkezi ise alt kromosferik ısınmayı doğrudan yansıtır."
    },
    "Demir I (Fe I - 6302.5 Å) [Manyetik Hassas]": {
        "element": "Nötr Demir (Fe I)",
        "lambda0": 6302.5,
        "sigma": 0.07,
        "gamma": 0.05,
        "depth": 0.68,
        "g_eff": 2.50,
        "katman": "Alt / Orta Fotosfer",
        "gecis": "Manyetik Hassas Geçiş (g = 2.5)",
        "renk": "#38bdf8",
        "flare_sens": 0.20,
        "sdo_default": "HMI Manyetogram (Fotosferik Manyetik Alan Kutupları)",
        "aciklama": "Yüksek Landé faktörü (g = 2.5) sayesinde Güneş lekelerinin manyetik alan şiddetini ölçmek için uzay teleskoplarında (Hinode, SDO/HMI) standart olarak kullanılır. Güçlü manyetik alanda çizgi Zeeman etkisiyle belirgin biçimde ikiye/üçe yarılır."
    },
    "Hidrojen H-Beta (Hβ - 4861.3 Å)": {
        "element": "Hidrojen (H I)",
        "lambda0": 4861.3,
        "sigma": 0.28,
        "gamma": 0.35,
        "depth": 0.78,
        "g_eff": 1.00,
        "katman": "Alt / Orta Kromosfer",
        "gecis": "Balmer n=4 → n=2",
        "renk": "#06b6d4",
        "flare_sens": 0.95,
        "sdo_default": "AIA 304 Å (Kromosfer & Prominens - 50,000 K)",
        "aciklama": "Mavi-yeşil bölgedeki Balmer çizgisidir. H-Alfa'ya göre biraz daha derin atmosfer katmanlarını (alt kromosfer) tarar ve beyaz ışıklı patlamalara (White-Light Flares) duyarlıdır."
    }
}

# NASA SDO (Solar Dynamics Observatory) Canlı Görüntü Kanalları
SDO_CHANNELS = {
    "AIA 304 Å (Kromosfer & Prominens - 50,000 K)": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0304.jpg",
    "AIA 193 Å (Korona & Patlama Halkaları - 1.2 Milyon K)": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0193.jpg",
    "AIA 131 Å (Şiddetli Flare Plazması - 10 Milyon K)": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0131.jpg",
    "AIA 171 Å (Sakin Korona & Manyetik Arklar - 600,000 K)": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0171.jpg",
    "AIA 1600 Å (Üst Fotosfer & Geçiş Bölgesi - 10,000 K)": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_1600.jpg",
    "AIA 1700 Å (Üst Fotosfer & Minimum Sıcaklık - 4,500 K)": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_1700.jpg",
    "HMI Manyetogram (Fotosferik Manyetik Alan Kutupları)": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMIBC.jpg",
    "HMI Sürekli Işınım (Görünür Işık & Güneş Lekeleri)": "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_HMIIC.jpg"
}

# -------------------------------------------------------------------------
# 4. CANLI NOAA GOES VERİ ÇEKME FONKSİYONU
# -------------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_noaa_live_data():
    url_xray = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
    url_mag = "https://services.swpc.noaa.gov/json/goes/primary/magnetometers-6-hour.json"
    
    df_xray, df_mag = pd.DataFrame(), pd.DataFrame()
    hata_mesaji = None

    try:
        req_x = urllib.request.Request(url_xray, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_x, timeout=10) as resp:
            data_x = json.loads(resp.read().decode())
        df_x = pd.DataFrame(data_x)
        df_xray = df_x[df_x['energy'] == '0.1-0.8nm'].copy()
        df_xray['time_tag'] = pd.to_datetime(df_xray['time_tag'])
        df_xray['flux'] = pd.to_numeric(df_xray['flux'], errors='coerce')
        df_xray = df_xray.dropna(subset=['flux']).sort_values('time_tag')
    except Exception as e:
        hata_mesaji = f"NOAA X-Ray bağlantı uyarısı: {e}"

    try:
        req_m = urllib.request.Request(url_mag, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req_m, timeout=10) as resp:
            data_m = json.loads(resp.read().decode())
        df_m = pd.DataFrame(data_m)
        df_mag = df_m.copy()
        df_mag['time_tag'] = pd.to_datetime(df_mag['time_tag'])
        df_mag['Hp'] = pd.to_numeric(df_mag['Hp'], errors='coerce')
        df_mag = df_mag.dropna(subset=['Hp']).sort_values('time_tag')
    except Exception as e:
        hata_mesaji = f"NOAA Manyetometre bağlantı uyarısı: {e}"

    return df_xray, df_mag, hata_mesaji

def classify_flare(flux_val):
    if flux_val is None or np.isnan(flux_val) or flux_val <= 0:
        return "Veri Yok", "⚪", "#9ca3af"
    if flux_val < 1e-7:
        sub = flux_val / 1e-8
        return f"A{sub:.1f}", "🟢 SAKİN (A-Sınıfı)", "#10b981"
    elif flux_val < 1e-6:
        sub = flux_val / 1e-7
        return f"B{sub:.1f}", "🟢 NORMAL (B-Sınıfı)", "#22c55e"
    elif flux_val < 1e-5:
        sub = flux_val / 1e-6
        return f"C{sub:.1f}", "🟡 AKTİF (C-Sınıfı Küçük Patlama)", "#eab308"
    elif flux_val < 1e-4:
        sub = flux_val / 1e-5
        return f"M{sub:.1f}", "🟠 ORTA ŞİDDETLİ PATLAMA (M-Sınıfı)", "#f97316"
    else:
        sub = flux_val / 1e-4
        return f"X{sub:.1f}", "🔴 EKSTREM PATLAMA (X-Sınıfı)", "#ef4444"

# -------------------------------------------------------------------------
# 5. SOL MENÜ (KONTROL PANELİ)
# -------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.title("🔬 Spektroskopi & Filtre
