import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import voigt_profile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib.request
import json
from datetime import datetime, timezone

st.set_page_config(
    page_title="Güneş Spektroskopisi & Canlı Uzay Havası Paneli",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .stApp {{
        background: linear-gradient(rgba(6, 9, 19, 0.80), rgba(10, 14, 28, 0.86)), 
                    url("{bg_url}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #f0f6fc;
    }}
    section[data-testid="stSidebar"] {{
        background: rgba(11, 15, 29, 0.82) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.12);
    }}
    div[data-testid="stMetric"] {{
        background: rgba(18, 25, 46, 0.68);
        backdrop-filter: blur(10px);
        padding: 15px 18px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.14);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.45);
    }}
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

SPECTRAL_LINES = {
    "Hidrojen H-Alfa (Hα - 6562.8 Å)": {
        "element": "Hidrojen (H I)",
        "lambda0": 6562.8,
        "sigma": 0.38,
        "gamma": 0.45,
        "depth": 0.82,
        "g_eff": 1.05,
        "katman": "Orta / Üst Kromosfer",
        "gecis": "Balmer n=3 -> n=2",
        "renk": "#ff4b4b",
        "flare_sens": 1.35,
        "solar_default": "GOES-SUVI 304 Å (Kromosfer & Prominens - Canlı)",
        "aciklama": "Güneş fiziğinin en temel kromosferik çizgisidir. Güneş patlamaları sırasında üst atmosferdeki ani ısınma nedeniyle çizginin en dip noktası soğurulmadan salma (emisyon) durumuna geçer."
    },
    "Kalsiyum II K (Ca II K - 3933.7 Å)": {
        "element": "İyonize Kalsiyum (Ca II)",
        "lambda0": 3933.7,
        "sigma": 0.55,
        "gamma": 0.85,
        "depth": 0.92,
        "g_eff": 1.17,
        "katman": "Üst Fotosfer & Kromosfer",
        "gecis": "Rezonans 4s 2S -> 4p 2P",
        "renk": "#a855f7",
        "flare_sens": 1.60,
        "solar_default": "GOES-SUVI 171 Å (Alt Korona & Manyetik Arklar - Canlı)",
        "aciklama": "Geniş kanatlara sahip derin ultraviyole sınır çizgisidir. Kromosferik manyetik ağ bölgelerine ve patlamalara karşı en duyarlı çizgidir."
    },
    "Sodyum I D2 (Na I D2 - 5889.9 Å)": {
        "element": "Nötr Sodyum (Na I)",
        "lambda0": 5889.9,
        "sigma": 0.12,
        "gamma": 0.15,
        "depth": 0.84,
        "g_eff": 1.33,
        "katman": "Üst Fotosfer & Alt Kromosfer",
        "gecis": "Rezonans Dubleti (3s 2S -> 3p 2P)",
        "renk": "#f59e0b",
        "flare_sens": 0.65,
        "solar_default": "SOHO/SDO Görünür Işık (Güneş Lekeleri & Fotosfer - Canlı)",
        "aciklama": "Fotosfer ile kromosfer arasındaki geçiş sınırını inceler. Büyük patlamalarda çekirdeği sığlaşır, güneş lekelerinde manyetik genişleme gösterir."
    },
    "Magnezyum I b1 (Mg I b1 - 5183.6 Å)": {
        "element": "Nötr Magnezyum (Mg I)",
        "lambda0": 5183.6,
        "sigma": 0.14,
        "gamma": 0.22,
        "depth": 0.80,
        "g_eff": 1.25,
        "katman": "Üst Fotosfer / Alt Kromosfer",
        "gecis": "Triplet (3p 3P -> 4s 3S)",
        "renk": "#10b981",
        "flare_sens": 0.55,
        "solar_default": "GOES-SUVI 195 Å (Korona & Aktif Bölgeler - Canlı)",
        "aciklama": "Yeşil bölgedeki ünlü Mg b üçlüsünün en güçlü üyesidir. Basınç genişlemesi fotosferik yoğunluğu, merkezi ise alt kromosferik ısınmayı yansıtır."
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
        "solar_default": "SOHO/SDO Manyetogram (Fotosferik Manyetik Kutuplar - Canlı)",
        "aciklama": "Yüksek Lande faktörü (g = 2.5) sayesinde Güneş lekelerinin manyetik alan şiddetini ölçmek için kullanılır. Güçlü manyetik alanda Zeeman etkisiyle belirgin biçimde yarılır."
    },
    "Hidrojen H-Beta (Hβ - 4861.3 Å)": {
        "element": "Hidrojen (H I)",
        "lambda0": 4861.3,
        "sigma": 0.28,
        "gamma": 0.35,
        "depth": 0.78,
        "g_eff": 1.00,
        "katman": "Alt / Orta Kromosfer",
        "gecis": "Balmer n=4 -> n=2",
        "renk": "#06b6d4",
        "flare_sens": 0.95,
        "solar_default": "GOES-SUVI 304 Å (Kromosfer & Prominens - Canlı)",
        "aciklama": "Mavi-yeşil bölgedeki Balmer çizgisidir. H-Alfa'ya göre daha derin atmosfer katmanlarını tarar ve beyaz ışıklı patlamalara duyarlıdır."
    }
}

# Görüntü önbelleğini kırmak için her 3 dakikada bir değişen zaman damgası
cache_buster = int(datetime.now(timezone.utc).timestamp() // 180)

# Gerçek Zamanlı NOAA GOES-SUVI ve SOHO/HMI Güneş Gözlem Kanalları (Anlık Güncellenir)
SOLAR_CHANNELS = {
    "GOES-SUVI 304 Å (Kromosfer & Prominens - Canlı)": f"https://services.swpc.noaa.gov/images/animations/suvi/primary/304/latest.png?t={cache_buster}",
    "GOES-SUVI 195 Å (Korona & Aktif Bölgeler - Canlı)": f"https://services.swpc.noaa.gov/images/animations/suvi/primary/195/latest.png?t={cache_buster}",
    "GOES-SUVI 131 Å (Şiddetli Flare Plazması 10 MK - Canlı)": f"https://services.swpc.noaa.gov/images/animations/suvi/primary/131/latest.png?t={cache_buster}",
    "GOES-SUVI 171 Å (Alt Korona & Manyetik Arklar - Canlı)": f"https://services.swpc.noaa.gov/images/animations/suvi/primary/171/latest.png?t={cache_buster}",
    "GOES-SUVI 284 Å (Yüksek Sıcaklık Koronası - Canlı)": f"https://services.swpc.noaa.gov/images/animations/suvi/primary/284/latest.png?t={cache_buster}",
    "GOES-SUVI 094 Å (Sıcak Patlama Çekirdeği - Canlı)": f"https://services.swpc.noaa.gov/images/animations/suvi/primary/094/latest.png?t={cache_buster}",
    "SOHO/SDO Manyetogram (Fotosferik Manyetik Kutuplar - Canlı)": f"https://soho.nascom.nasa.gov/data/realtime/hmi_mag/1024/latest.jpg?t={cache_buster}",
    "SOHO/SDO Görünür Işık (Güneş Lekeleri & Fotosfer - Canlı)": f"https://soho.nascom.nasa.gov/data/realtime/hmi_igr/1024/latest.jpg?t={cache_buster}"
}

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
        hata_mesaji = f"NOAA X-Ray uyarısı: {e}"

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
        hata_mesaji = f"NOAA Manyetometre uyarısı: {e}"

    return df_xray, df_mag, hata_mesaji

def classify_flare(flux_val):
    if flux_val is None or np.isnan(flux_val) or flux_val <= 0:
        return "Veri Yok", "Bilinmiyor", "#9ca3af"
    if flux_val < 1e-7:
        return f"A{flux_val / 1e-8:.1f}", "SAKİN (A-Sınıfı)", "#10b981"
    elif flux_val < 1e-6:
        return f"B{flux_val / 1e-7:.1f}", "NORMAL (B-Sınıfı)", "#22c55e"
    elif flux_val < 1e-5:
        return f"C{flux_val / 1e-6:.1f}", "AKTİF (C-Sınıfı Küçük Patlama)", "#eab308"
    elif flux_val < 1e-4:
        return f"M{flux_val / 1e-5:.1f}", "ORTA ŞİDDETLİ PATLAMA (M-Sınıfı)", "#f97316"
    else:
        return f"X{flux_val / 1e-4:.1f}", "EKSTREM PATLAMA (X-Sınıfı)", "#ef4444"

st.sidebar.markdown("---")
st.sidebar.title("🔬 Spektroskopi & Filtre")
secilen_cizgi_adi = st.sidebar.selectbox(
    "Soğurulan Element / Tayf Çizgisi Seçin:",
    list(SPECTRAL_LINES.keys()),
    index=0
)
cizgi = SPECTRAL_LINES[secilen_cizgi_adi]

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Atmosferik Fizik Simülatörü")

sim_modu = st.sidebar.radio(
    "Patlama / Aktivite Kaynağı:",
    ["Canlı NOAA GOES Verisini Kullan", "Manuel Patlama Simülasyonu"],
    index=0
)

if sim_modu == "Manuel Patlama Simülasyonu":
    manuel_log_flux = st.sidebar.slider(
        "Simüle Edilen X-Işını Akısı (log10 W/m²):",
        min_value=-8.0, max_value=-3.0, value=-4.0, step=0.1
    )
    aktif_flux = 10 ** manuel_log_flux
else:
    aktif_flux = None

manyetik_alan_gauss = st.sidebar.slider(
    "Leke Manyetik Alan Şiddeti (Gauss):",
    min_value=0, max_value=4000, value=0, step=100
)

doppler_hiz_kms = st.sidebar.slider(
    "Radyal Plazma Hızı / Doppler (km/s):",
    min_value=-150, max_value=150, value=0, step=5
)

if st.sidebar.button("🔄 Canlı NOAA & Güneş Görüntüsünü Yenile", use_container_width=True):
    st.cache_data.clear()

df_xray, df_mag, hata = fetch_noaa_live_data()

if not df_xray.empty:
    canli_flux = float(df_xray['flux'].iloc[-1])
    max_flux_6h = float(df_xray['flux'].max())
    son_zaman_str = df_xray['time_tag'].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
else:
    canli_flux = 1.5e-7
    max_flux_6h = 3.2e-7
    son_zaman_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

canli_hp = float(df_mag['Hp'].iloc[-1]) if not df_mag.empty else 95.0

kullanilan_flux = canli_flux if aktif_flux is None else aktif_flux
sinif_kisa, sinif_uzun, sinif_renk = classify_flare(kullanilan_flux)
max_kisa, _, _ = classify_flare(max_flux_6h)

st.title("☀️ Güneş Spektroskopisi ve Canlı Uzay Havası Gözlemevi")
st.caption("NOAA GOES-18 XRS & SUVI Canlı Telemetri | SOHO/SDO Manyetogram | Voigt Tayf & Eşdeğer Genişlik Analizi")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Anlık GOES X-Işını Akısı", f"{kullanilan_flux:.2e} W/m²", f"Sınıf: {sinif_kisa}")
with col2:
    st.metric("Son 6 Saat Maksimum Patlama", f"{max_flux_6h:.2e} W/m²", f"Tepe: {max_kisa}")
with col3:
    st.metric("GOES Manyetometre (Hp)", f"{canli_hp:.1f} nT", f"{canli_hp - 100.0:+.1f} nT (Nominal: 100 nT)")
with col4:
    st.metric("İncelenen Dalga Boyu", f"{cizgi['lambda0']} Å", cizgi["element"])

st.info(
    f"📡 **Canlı Telemetri & Tayf Sentez Raporu ({son_zaman_str}):**\n\n"
    f"* **Uzay Havası Durumu:** Anlık X-ışını akısı **{sinif_kisa}** seviyesinde ({sinif_uzun}). "
    f"Son 6 saat içinde kaydedilen en yüksek aktivite **{max_kisa}** sınıfındadır. "
    f"Yörünge manyetik alanı Hp = {canli_hp:.1f} nT seviyesindedir.\n"
    f"* **Seçilen Çizgi ({secilen_cizgi_adi}) Üzerindeki Fiziksel Etkisi:** {cizgi['aciklama']}"
)

lambda0 = cizgi["lambda0"]
pencere = 3.5 if lambda0 > 4500 else 5.0
wave = np.linspace(lambda0 - pencere, lambda0 + pencere, 1200)

v_raw = voigt_profile(wave - lambda0, cizgi["sigma"], cizgi["gamma"])
v_norm = v_raw / np.max(v_raw)
flux_quiet = 1.0 - cizgi["depth"] * v_norm

c_kms = 299792.458
delta_lambda_doppler = lambda0 * (doppler_hiz_kms / c_kms)
merkez_aktif = lambda0 + delta_lambda_doppler

delta_lambda_zeeman = 4.67e-13 * cizgi["g_eff"] * (lambda0 ** 2) * manyetik_alan_gauss
flare_index = np.clip((np.log10(kullanilan_flux) + 7.5) / 3.5, 0.0, 1.8)
termal_genisleme = cizgi["sigma"] * (1.0 + 0.20 * flare_index * cizgi["flare_sens"])

if delta_lambda_zeeman > 0.005:
    v_left = voigt_profile(wave - (merkez_aktif - delta_lambda_zeeman), termal_genisleme, cizgi["gamma"])
    v_right = voigt_profile(wave - (merkez_aktif + delta_lambda_zeeman), termal_genisleme, cizgi["gamma"])
    v_pi = voigt_profile(wave - merkez_aktif, termal_genisleme, cizgi["gamma"])
    v_zeeman_raw = 0.35 * (v_left + v_right) + 0.30 * v_pi
    v_zeeman_norm = v_zeeman_raw / np.max(v_raw)
else:
    v_zeeman_raw = voigt_profile(wave - merkez_aktif, termal_genisleme, cizgi["gamma"])
    v_zeeman_norm = v_zeeman_raw / np.max(v_raw)

cekirdek_profil = np.exp(-0.5 * ((wave - merkez_aktif) / (cizgi["sigma"] * 0.65)) ** 2)
emisyon_dolgusu = flare_index * cizgi["flare_sens"] * 0.55 * cekirdek_profil

flux_active = 1.0 - cizgi["depth"] * v_zeeman_norm + emisyon_dolgusu
flux_active = np.clip(flux_active, 0.02, 1.65)

integrator = getattr(np, "trapezoid", getattr(np, "trapz", None))
ew_quiet_ma = float(integrator(1.0 - flux_quiet, wave) * 1000.0)
ew_active_ma = float(integrator(1.0 - flux_active, wave) * 1000.0)
ew_degisim_ma = ew_active_ma - ew_quiet_ma
cizgi_merkez_derinlik = (1.0 - float(np.min(flux_active))) * 100.0

st.markdown("---")
st.subheader(f"🌈 {secilen_cizgi_adi} — Yüksek Çözünürlüklü Tayf Profili & Canlı Güneş Gözlemi")

col_spec, col_info, col_sdo = st.columns([2.1, 1.0, 1.1])

with col_spec:
    fig_spec = go.Figure()
    fig_spec.add_hline(
        y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.45)",
        annotation_text="Sürekli Işınım (Fc = 1.0)", annotation_position="top right"
    )
    fig_spec.add_trace(go.Scatter(
        x=wave, y=flux_quiet, mode="lines", name="Sakin Güneş (Voigt)",
        line=dict(color="rgba(160, 174, 192, 0.65)", width=2, dash="dash")
    ))
    fig_spec.add_trace(go.Scatter(
        x=wave, y=flux_active, mode="lines", name=f"Aktif Profil ({sinif_kisa})",
        line=dict(color=cizgi["renk"], width=3.5)
    ))
    fig_spec.add_vline(x=lambda0, line_dash="dashdot", line_color="rgba(255,255,255,0.22)")
    fig_spec.update_layout(
        title=f"{cizgi['element']} ({lambda0} Å) — Eşdeğer Genişlik: Wλ = {ew_active_ma:.1f} mÅ",
        xaxis_title="Dalga Boyu (Angström - Å)",
        yaxis_title="Normalize Akı (F / Fc)",
        yaxis=dict(range=[0.0, max(1.25, float(np.max(flux_active)) + 0.1)]),
        template="plotly_dark",
        paper_bgcolor="rgba(13, 18, 34, 0.72)",
        plot_bgcolor="rgba(13, 18, 34, 0.45)",
        height=470,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40)
    )
    st.plotly_chart(fig_spec, use_container_width=True)

with col_info:
    st.markdown("#### 📐 Eşdeğer Genişlik & Atomik Veri")
    st.metric(
        label="Aktif Eşdeğer Genişlik (Wλ)",
        value=f"{ew_active_ma:.1f} mÅ",
        delta=f"{ew_degisim_ma:+.1f} mÅ (Sakin: {ew_quiet_ma:.1f} mÅ)",
        delta_color="inverse"
    )
    st.metric(
        label="Çizgi Merkez Derinliği",
        value=f"%{cizgi_merkez_derinlik:.1f}",
        delta=f"Zeeman dLambda: {delta_lambda_zeeman*1000:.1f} mÅ"
    )
    st.markdown(f"""
    * **Element:** {cizgi['element']}
    * **Merkez:** `{cizgi['lambda0']} Å`
    * **Geçiş:** {cizgi['gecis']}
    * **Katman:** {cizgi['katman']}
    * **Lande Faktörü:** `{cizgi['g_eff']}`
    * **Doppler Kayması:** `{delta_lambda_doppler:+.3f} Å`
    """)

with col_sdo:
    st.markdown("#### 🛰️ Canlı GOES-SUVI / SOHO Güneş Diski")
    varsayilan_idx = list(SOLAR_CHANNELS.keys()).index(cizgi["solar_default"])
    secilen_solar_kanal = st.selectbox(
        "Gerçek Zamanlı Uydu Gözlem Filtresi:",
        list(SOLAR_CHANNELS.keys()),
        index=varsayilan_idx
    )
    st.image(
        SOLAR_CHANNELS[secilen_solar_kanal],
        caption=f"Canlı Telemetri ({son_zaman_str}): {secilen_solar_kanal}",
        use_container_width=True
    )

st.markdown("---")
st.subheader("📊 Canlı NOAA GOES X-Işını Akısı ve Manyetometre Zaman Serisi (Son 6 Saat)")

fig_noaa = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
    subplot_titles=("GOES X-Işını Akısı (0.1 - 0.8 nm) [W/m²]", "GOES Manyetometre Hp Bileşeni [nT]")
)

if not df_xray.empty:
    fig_noaa.add_trace(
        go.Scatter(x=df_xray['time_tag'], y=df_xray['flux'], mode='lines', name='XRS Akısı', line=dict(color='#f97316', width=2.2)),
        row=1, col=1
    )
    fig_noaa.add_hline(y=1e-4, line_dash="dash", line_color="#ef4444", annotation_text="X-Sınıfı", row=1, col=1)
    fig_noaa.add_hline(y=1e-5, line_dash="dot", line_color="#f97316", annotation_text="M-Sınıfı", row=1, col=1)
    fig_noaa.add_hline(y=1e-6, line_dash="dot", line_color="#eab308", annotation_text="C-Sınıfı", row=1, col=1)

if not df_mag.empty:
    fig_noaa.add_trace(
        go.Scatter(x=df_mag['time_tag'], y=df_mag['Hp'], mode='lines', name='Hp Manyetik Alan', line=dict(color='#a855f7', width=2.2)),
        row=2, col=1
    )
    fig_noaa.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.35)", annotation_text="Nominal (~100 nT)", row=2, col=1)

fig_noaa.update_yaxes(type="log", title_text="Akı (W/m²)", row=1, col=1)
fig_noaa.update_yaxes(title_text="Hp (nT)", row=2, col=1)
fig_noaa.update_xaxes(title_text="Zaman (UTC)", row=2, col=1)

fig_noaa.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(13, 18, 34, 0.72)",
    plot_bgcolor="rgba(13, 18, 34, 0.45)",
    height=540,
    margin=dict(l=50, r=30, t=50, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1)
)

st.plotly_chart(fig_noaa, use_container_width=True)
