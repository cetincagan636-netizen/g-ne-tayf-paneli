import streamlit as st
import json
import urllib.request
import pandas as pd
import numpy as np
from scipy.special import voigt_profile
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Güneş Spektroskopisi & Canlı Uzay Havası İstasyonu",
    page_icon="☀️",
    layout="wide"
)

# --- ELEMENT VE TAYF ÇİZGİSİ VERİTABANI ---
TAYF_VERITABANI = {
    "Hidrojen H-Alfa (Hα - 6562.8 Å)": {
        "element": "Hidrojen (H I)",
        "dalgaboyu": 6562.8,
        "gecis": "Balmer Serisi (n = 3 → 2)",
        "katman": "Üst Fotosfer (Kanatlar) & Orta/Üst Kromosfer (Çekirdek)",
        "renk": "#ff3333",
        "sigma": 0.42,   # Termal Doppler genişliği
        "gamma": 0.55,   # Stark / Basınç genişliği (kanatlar)
        "derinlik": 0.82,
        "kromosferik_duyarlilik": 1.0,  # Patlamalara çok yüksek tepki (emisyon tersinmesi)
        "bilgi": "Güneş fiziğinin en temel kromosferik çizgisidir. Güneş patlamaları (Flare) sırasında üst atmosferdeki ani ısınma ve elektron bombardımanı nedeniyle çizginin en dip noktası (çekirdeği) soğurulmadan salma (emisyon) durumuna geçer ve çukur yukarı doğru dolar."
    },
    "Hidrojen H-Beta (Hβ - 4861.3 Å)": {
        "element": "Hidrojen (H I)",
        "dalgaboyu": 4861.3,
        "gecis": "Balmer Serisi (n = 4 → 2)",
        "katman": "Orta ve Üst Fotosfer (LTE Baskın)",
        "renk": "#00ccff",
        "sigma": 0.32,
        "gamma": 0.45,
        "derinlik": 0.75,
        "kromosferik_duyarlilik": 0.45,
        "bilgi": "Hα'ya göre osilatör şiddeti yaklaşık 5.4 kat daha düşüktür. Bu sayede fotonlar daha derin ve dengeli fotosfer katmanlarından gelir. Yıldızların yüzey sıcaklığı (Teff) ve yüzey çekimi (log g) ölçümünde birincil standarttır."
    },
    "Kalsiyum II K (Ca II K - 3933.7 Å)": {
        "element": "İyonize Kalsiyum (Ca II)",
        "dalgaboyu": 3933.7,
        "gecis": "Rezonans Geçişi (4s ²S → 4p ²P)",
        "katman": "Alt ve Orta Kromosfer (Manyetik Plage Bölgeleri)",
        "renk": "#9933ff",
        "sigma": 0.55,
        "gamma": 1.10,
        "derinlik": 0.90,
        "kromosferik_duyarlilik": 1.2,
        "bilgi": "Güneş tayfındaki en geniş ve en derin rezonans çizgilerinden biridir. Manyetik aktif bölgelere (plage) ve kromosferik ağ yapısına aşırı duyarlıdır. Çekirdeğindeki K2v ve K2r emisyon tepeleri doğrudan yüzey manyetik alan şiddetini ele verir."
    },
    "Magnezyum I b1 (Mg I b - 5183.6 Å)": {
        "element": "Nötr Magnezyum (Mg I)",
        "dalgaboyu": 5183.6,
        "gecis": "Triplet Geçişi (3p ³P → 4s ³S)",
        "katman": "Üst Fotosfer / Sıcaklık Minimumu Bölgesi",
        "renk": "#00cc66",
        "sigma": 0.14,
        "gamma": 0.18,
        "derinlik": 0.72,
        "kromosferik_duyarlilik": 0.25,
        "bilgi": "Fotosfer ile kromosfer arasındaki sınır katmanı (sıcaklık minimumu ~4200 K) incelemek ve yüzey yerçekimi ivmesini (log g) yüksek hassasiyetle belirlemek için kullanılır."
    },
    "Sodyum I D2 (Na I D2 - 5889.9 Å)": {
        "element": "Nötr Sodyum (Na I)",
        "dalgaboyu": 5889.9,
        "gecis": "Rezonans Dubleti (3s ²S → 3p ²P)",
        "katman": "Üst Fotosfer & Alt Kromosfer",
        "renk": "#ffaa00",
        "sigma": 0.12,
        "gamma": 0.15,
        "derinlik": 0.85,
        "kromosferik_duyarlilik": 0.30,
        "bilgi": "Sarı-turuncu bölgedeki meşhur Sodyum D çiftinin güçlü bileşenidir. Fotosferik hız alanları (Doppler kaymaları) ve güneş lekelerinin üzerindeki gaz hareketlerini ölçmekte kullanılır."
    },
    "Demir I (Fe I - 6302.5 Å)": {
        "element": "Nötr Demir (Fe I)",
        "dalgaboyu": 6302.5,
        "gecis": "Manyetik Duyarlı Geçiş (Landé g = 2.5)",
        "katman": "Derin Fotosfer (Leke & Manyetogram Kaynağı)",
        "renk": "#cc6600",
        "sigma": 0.08,
        "gamma": 0.04,
        "derinlik": 0.62,
        "kromosferik_duyarlilik": 0.05,
        "bilgi": "Stark kanatları neredeyse hiç yoktur, çok dar ve keskin bir fotosferik çukurdur. Landé faktörü (g = 2.5) çok yüksek olduğu için manyetik alan altında Zeeman yarılmasına uğrar; SDO/HMI ve yer konuşlu manyetogramların temel çalışma prensibidir."
    }
}

# --- CANLI NOAA VERİ ÇEKME FONKSİYONU ---
@st.cache_data(ttl=60)
def noaa_veri_cek():
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # 1. GOES X-Ray
    url_xray = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
    req_x = urllib.request.Request(url_xray, headers=headers)
    with urllib.request.urlopen(req_x, timeout=10) as r:
        df_xray = pd.DataFrame(json.loads(r.read().decode()))
    df_xray = df_xray[df_xray['energy'] == '0.1-0.8nm'].copy()
    df_xray['time_tag'] = pd.to_datetime(df_xray['time_tag'])
    df_xray['flux'] = pd.to_numeric(df_xray['flux'], errors='coerce')
    df_xray = df_xray.dropna(subset=['flux'])

    # 2. GOES Manyetometre
    url_mag = "https://services.swpc.noaa.gov/json/goes/primary/magnetometers-6-hour.json"
    req_m = urllib.request.Request(url_mag, headers=headers)
    with urllib.request.urlopen(req_m, timeout=10) as r:
        df_mag = pd.DataFrame(json.loads(r.read().decode()))
    df_mag['time_tag'] = pd.to_datetime(df_mag['time_tag'])
    df_mag['Hp'] = pd.to_numeric(df_mag['Hp'], errors='coerce')
    df_mag = df_mag.dropna(subset=['Hp'])

    return df_xray, df_mag

def patlama_sinifi_bul(aki):
    if aki >= 1e-4:
        return f"X{aki/1e-4:.1f}", "🔴 KRİTİK (X-Sınıfı Dev Patlama)"
    elif aki >= 1e-5:
        return f"M{aki/1e-5:.1f}", "🟠 YÜKSEK (M-Sınıfı Orta/Güçlü Patlama)"
    elif aki >= 1e-6:
        return f"C{aki/1e-6:.1f}", "🟡 AKTİF (C-Sınıfı Küçük Patlama)"
    elif aki >= 1e-7:
        return f"B{aki/1e-7:.1f}", "🟢 SAKİN (B-Sınıfı Arka Plan)"
    else:
        return f"A{aki/1e-8:.1f}", "🔵 ÇOK SAKİN (A-Sınıfı Minimum)"

# --- KENAR ÇUBUĞU (SIDEBAR) KONTROLLERİ ---
st.sidebar.title("🔬 Spektroskopi & Filtre")
secilen_cizgi_adi = st.sidebar.selectbox(
    "Soğurulan Element / Tayf Çizgisi Seçin:",
    list(TAYF_VERITABANI.keys())
)
cizgi = TAYF_VERITABANI[secilen_cizgi_adi]

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Atmosferik Fizik Simülatörü")
sim_modu = st.sidebar.radio(
    "Patlama / Aktivite Kaynağı:",
    ["Canlı NOAA GOES Verisini Kullan", "Manuel Patlama Simülasyonu"]
)

if sim_modu == "Manuel Patlama Simülasyonu":
    manuel_log_aki = st.sidebar.slider(
        "Simüle Edilen X-Işını Akısı (log10 W/m²):",
        min_value=-7.5, max_value=-3.5, value=-5.0, step=0.1,
        help="-7: B Sınıfı | -6: C Sınıfı | -5: M Sınıfı | -4: X Sınıfı"
    )
    aktif_aki_sim = 10 ** manuel_log_aki
else:
    aktif_aki_sim = None

zeeman_b = st.sidebar.slider(
    "Leke Manyetik Alan Şiddeti (Gauss):",
    min_value=0, max_value=3500, value=0, step=100,
    help="Özellikle Fe I (6302.5 Å) çizgisinde Zeeman yarılmasını ve genişlemesini gösterir."
)

if st.sidebar.button("🔄 Canlı NOAA Verisini Şimdi Yenile"):
    st.cache_data.clear()

# --- ANA EKRAN BAŞLIĞI ---
st.title("☀️ Canlı Uzay Havası & Güneş Spektroskopisi Analiz Platformu")
st.markdown("NOAA GOES uydusundan gelen gerçek zamanlı X-ışını ve manyetometre telemetrisi ile seçilen elementin **Voigt soğurulma profilini** eşzamanlı inceler.")

# Verileri Çek
try:
    df_xray, df_mag = noaa_veri_cek()
    anlik_aki = df_xray['flux'].iloc[-1]
    max_aki_6h = df_xray['flux'].max()
    anlik_hp = df_mag['Hp'].iloc[-1]
    son_zaman = df_xray['time_tag'].iloc[-1].strftime("%Y-%m-%d %H:%M UTC")
except Exception as e:
    st.error(f"NOAA bağlantı hatası: {e}")
    st.stop()

kullanilan_aki = aktif_aki_sim if aktif_aki_sim is not None else anlik_aki
sinif_kodu, sinif_durum = patlama_sinifi_bul(kullanilan_aki)
max_sinif_kodu, _ = patlama_sinifi_bul(max_aki_6h)

# --- ÜST METRİK KARTLARI ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Anlık GOES X-Işını Akısı", f"{anlik_aki:.2e} W/m²", f"Sınıf: {sinif_kodu}")
col2.metric("Son 6 Saat Maksimum Patlama", f"{max_aki_6h:.2e} W/m²", f"Tepe: {max_sinif_kodu}")
col3.metric("GOES Manyetometre (Hp)", f"{anlik_hp:.1f} nT", f"{anlik_hp - 100:+.1f} nT (Nominal: 100 nT)")
col4.metric("İncelenen Dalga Boyu", f"{cizgi['dalgaboyu']} Å", cizgi['element'])

# --- OTOMATİK BİLİMSEL YORUM MOTORU ---
st.info(
    f"**📡 Canlı Telemetri & Tayf Sentez Raporu ({son_zaman}):**\n\n"
    f"* **Uzay Havası Durumu:** Anlık X-ışını akısı **{sinif_kodu}** seviyesinde ({sinif_durum}). "
    f"Son 6 saat içinde kaydedilen en yüksek aktivite **{max_sinif_kodu}** sınıfındadır. "
    f"Yörünge manyetik alanı $H_p = {anlik_hp:.1f}\\text{{ nT}}$ ile {'manyetosferik gündüz sıkışması (kompresyon) gösteriyor.' if anlik_hp > 110 else 'sakin/nominal seyrediyor.'}\n"
    f"* **Seçilen Çizgi ({secilen_cizgi_adi}) Üzerindeki Fiziksel Etkisi:** {cizgi['bilgi']}"
)

# --- BÖLÜM 1: CANLI NOAA TELEMETRİ GRAFİKLERİ (PLOTLY) ---
fig_noaa = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
    subplot_titles=("GOES X-Işını Akısı (0.1 - 0.8 nm) [W/m²]", "GOES Manyetometre Hp Bileşeni [nT]")
)

fig_noaa.add_trace(
    go.Scatter(x=df_xray['time_tag'], y=df_xray['flux'], mode='lines',
               name='XRS Akısı', line=dict(color='darkorange', width=2)),
    row=1, col=1
)
# Sınıf eşik çizgileri
fig_noaa.add_hline(y=1e-4, line_dash="dash", line_color="red", annotation_text="X Sınıfı", row=1, col=1)
fig_noaa.add_hline(y=1e-5, line_dash="dot", line_color="orangered", annotation_text="M Sınıfı", row=1, col=1)
fig_noaa.add_hline(y=1e-6, line_dash="dot", line_color="royalblue", annotation_text="C Sınıfı", row=1, col=1)

fig_noaa.add_trace(
    go.Scatter(x=df_mag['time_tag'], y=df_mag['Hp'], mode='lines',
               name='Hp Manyetik Alan', line=dict(color='mediumpurple', width=2)),
    row=2, col=1
)
fig_noaa.add_hline(y=100, line_dash="dash", line_color="gray", annotation_text="Nominal (~100 nT)", row=2, col=1)

fig_noaa.update_yaxes(type="log", range=[-7.5, -3.5], title_text="Akı (W/m²)", row=1, col=1)
fig_noaa.update_yaxes(title_text="Hp (nT)", row=2, col=1)
fig_noaa.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20), template="plotly_dark")

st.plotly_chart(fig_noaa, use_container_width=True)

# --- BÖLÜM 2: SEÇİLEN ELEMENTİN VOIGT TAYF PROFİLİ VE PATLAMA TEPKİSİ ---
st.markdown("---")
st.subheader(f"🌈 {secilen_cizgi_adi} — Yüksek Çözünürlüklü Tayf Profili ve Atmosferik Yanıt")

col_sol, col_sag = st.columns([2, 1])

with col_sol:
    merkez = cizgi['dalgaboyu']
    pencere = 12.0 if cizgi['gamma'] > 0.3 else 3.0
    dalga = np.linspace(merkez - pencere, merkez + pencere, 1000)
    delta_lam = dalga - merkez

    # Sakin Güneş (Quiet Sun) Voigt Profili
    v_ham = voigt_profile(delta_lam, cizgi['sigma'], cizgi['gamma'])
    v_norm = v_ham / np.max(v_ham)
    sakin_profil = 1.0 - cizgi['derinlik'] * v_norm

    # Patlama (Flare) ve Zeeman Etkisi Hesaplama
    # X-ray akısı 1e-7'nin üzerine çıktıkça kromosferik çekirdek ısınır ve emisyon yapar
    aktivite_faktoru = np.clip((np.log10(kullanilan_aki) + 7.0) / 3.5, 0.0, 1.2)
    emisyon_siddeti = aktivite_faktoru * cizgi['kromosferik_duyarlilik'] * 0.65

    # Zeeman yarılması (özellikle dar fotosferik çizgilerde belirgin)
    zeeman_kayma = (zeeman_b / 1000.0) * 0.08 * (2.5 if "Fe I" in secilen_cizgi_adi else 1.0)
    if zeeman_kayma > 0.01:
        v_sol = voigt_profile(delta_lam - zeeman_kayma, cizgi['sigma'], cizgi['gamma'])
        v_sag = voigt_profile(delta_lam + zeeman_kayma, cizgi['sigma'], cizgi['gamma'])
        v_birlesik = 0.5 * (v_sol + v_sag) / np.max(v_ham)
        baz_profil = 1.0 - cizgi['derinlik'] * v_birlesik
    else:
        baz_profil = sakin_profil.copy()

    # Kromosferik çekirdek emisyonu (Dar Gauss tepesi olarak çukurun dibini doldurur)
    cekirdek_emisyon = emisyon_siddeti * np.exp(-0.5 * (delta_lam / (cizgi['sigma'] * 0.65)) ** 2)
    aktif_profil = baz_profil + cekirdek_emisyon

    fig_tayf = go.Figure()
    fig_tayf.add_trace(go.Scatter(
        x=dalga, y=sakin_profil, mode='lines',
        name='Sakin Güneş (Referans Voigt Modeli)',
        line=dict(color='gray', width=2, dash='dash')
    ))
    fig_tayf.add_trace(go.Scatter(
        x=dalga, y=aktif_profil, mode='lines',
        name=f'Anlık / Aktif Profil ({sinif_kodu} Etkisi)',
        line=dict(color=cizgi['renk'], width=3)
    ))
    fig_tayf.add_hline(y=1.0, line_dash="dot", line_color="white", annotation_text="Sürekli Işınım (Continuum = 1.0)")

    fig_tayf.update_layout(
        title=f"{cizgi['element']} ({merkez} Å) Normalize Tayf Çizgisi Profili",
        xaxis_title="Dalga Boyu [Å (Angström)]",
        yaxis_title="Normalize Akı (F / Fc)",
        yaxis=dict(range=[0.0, 1.35]),
        height=420,
        template="plotly_dark"
    )
    st.plotly_chart(fig_tayf, use_container_width=True)

with col_sag:
    st.markdown("### 📋 Atomik ve Fiziksel Parametreler")
    st.write(f"**Element / İyon:** {cizgi['element']}")
    st.write(f"**Merkez Dalga Boyu ($\lambda_0$):** `{cizgi['dalgaboyu']} Å`")
    st.write(f"**Kuantum Geçişi:** {cizgi['gecis']}")
    st.write(f"**Oluşum Katmanı:** {cizgi['katman']}")
    st.write(f"**Termal Doppler ($\sigma$):** `{cizgi['sigma']} Å`")
    st.write(f"**Basınç / Stark ($\gamma$):** `{cizgi['gamma']} Å`")
    st.markdown("---")
    st.markdown("### 🔍 Nasıl Okunmalı?")
    st.markdown(
        "1. **Kesikli Gri Çizgi:** Güneş tamamen sakin olduğunda fotosfer ve kromosferin ürettiği standart soğurulma çukurudur.\n"
        "2. **Renkli Çizgi:** Canlı X-ışını akısına (veya sol menüden seçtiğin simülasyon şiddetine) göre çizginin çekirdeğinde oluşan **kromosferik ısınma / emisyon dolmasını** gösterir.\n"
        "3. Sol menüden **Manuel Patlama Simülasyonu**'nu seçip akıyı `-4.0` (X-Sınıfı) seviyesine çekersen $\\mathrm{H}\\alpha$ ve $\\mathrm{Ca\\ II\\ K}$ çekirdeklerinin nasıl emisyona geçtiğini, buna karşılık fotosferik $\\mathrm{Fe\\ I}$ çizgisinin neredeyse hiç etkilenmediğini canlı görebilirsin!"
    )
