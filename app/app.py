# app/app.py
# Multilingual UI (EN/AR/ES/FR/JA) + localized currency names + AUTO FX rates + optional override
# + hybrid risk + minutes + history + dev-menu toggle + RTL for Arabic

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests

# -------------------- i18n --------------------
TRANSLATIONS = {
    "en": {
        "app_title": "Credit Card Fraud Detector",
        "app_tagline": "Enter amount and time. Choose currency and unit — we’ll convert and score it. Clear verdict, confidence, and an audit-friendly history.",
        "settings": "Settings",
        "language": "Language",
        "hide_dev": "Hide Streamlit developer menu",
        "threshold": "Fraud threshold",
        "threshold_help": "If predicted fraud probability ≥ threshold, we'll flag as FRAUD.",
        "currency": "Currency",
        "fx_label": "Conversion rate → {sar_name}",
        "fx_caption": "Using **live** FX rates (auto-updated). You can override below if needed.",
        "fx_override": "Override live rate for this currency",
        "time_unit": "Time unit",
        "time_caption": "Time is converted to hours to match training (Time_Hours).",
        "amount_input": "Transaction Amount ({currency_name})",
        "time_input": "Time since first transaction ({unit})",
        "check_btn": "🔎 Check Transaction",
        "extreme_amount": "⚠️ This amount is unusually high compared to training data (over the 99th percentile).",
        "evaluating": "Evaluating…",
        "fraud_banner": "🚨 **FRAUDULENT transaction detected!**  (Confidence: {p:.2%})",
        "legit_banner": "✅ **Legitimate transaction.**  (Confidence: {p:.2%})",
        "risk_low": "Low risk",
        "risk_mid": "Mid risk",
        "risk_high": "High risk",
        "risk_card_meta": "Threshold: {thr:.2f} • Fraud probability: {p:.2%}",
        "confidence_title": "Prediction Confidence",
        "legitimate": "Legitimate",
        "fraudulent": "Fraudulent",
        "details_title": "Details used by the model",
        "history_title": "🗂️ Recent checks",
        "download_history": "⬇️ Download history (CSV)",
        "loading_model": "Loading model…",
        "model_shape_error": "Expected 28 V-features, got {n}. Retrain/export medians.",
        "lang_en": "English",
        "lang_ar": "Arabic",
        "lang_es": "Spanish",
        "lang_fr": "French",
        "lang_ja": "Japanese",
        "units": {"seconds":"seconds","minutes":"minutes","hours":"hours","days":"days","weeks":"weeks","months":"months","years":"years"},
        "currencies": {"USD":"US Dollar","SAR":"Saudi Riyal","EUR":"Euro","GBP":"Pound Sterling","JPY":"Japanese Yen","Custom":"Custom"},
        "sar_display": "Saudi Riyal (SAR)",
        "fx_status_ok": "Live FX rate in use",
        "fx_status_fallback": "Network issue — using fallback rates"
    },
    "ar": {
        "app_title": "كاشف الاحتيال في بطاقات الائتمان",
        "app_tagline": "أدخل المبلغ والوقت. اختر العملة ووحدة الزمن — سنحوّل ونقيّم العملية. نتيجة واضحة، نسبة ثقة، وسجل مناسب للتدقيق.",
        "settings": "الإعدادات",
        "language": "اللغة",
        "hide_dev": "إخفاء قائمة المطوّر في ستريملت",
        "threshold": "عتبة الاشتباه بالاحتيال",
        "threshold_help": "إذا كانت احتمالية الاحتيال ≥ العتبة، سيتم تصنيف العملية كاحتيال.",
        "currency": "العملة",
        "fx_label": "سعر التحويل ← {sar_name}",
        "fx_caption": "يتم استخدام أسعار صرف **مباشرة** (تُحدّث تلقائيًا). يمكنك التعديل يدويًا عند الحاجة.",
        "fx_override": "تجاوز السعر المباشر لهذه العملة",
        "time_unit": "وحدة الزمن",
        "time_caption": "يُحوَّل الزمن إلى ساعات ليتوافق مع التدريب (Time_Hours).",
        "amount_input": "قيمة العملية ({currency_name})",
        "time_input": "الوقت منذ أول عملية ({unit})",
        "check_btn": "🔎 فحص العملية",
        "extreme_amount": "⚠️ هذا المبلغ مرتفع للغاية مقارنةً ببيانات التدريب (فوق الشريحة 99%).",
        "evaluating": "جاري التقييم…",
        "fraud_banner": "🚨 **تم اكتشاف عملية احتيالية!** (نسبة الثقة: {p:.2%})",
        "legit_banner": "✅ **عملية سليمة.** (نسبة الثقة: {p:.2%})",
        "risk_low": "مخاطر منخفضة",
        "risk_mid": "مخاطر متوسطة",
        "risk_high": "مخاطر عالية",
        "risk_card_meta": "العتبة: {thr:.2f} • احتمال الاحتيال: {p:.2%}",
        "confidence_title": "مؤشر الثقة بالتنبؤ",
        "legitimate": "سليمة",
        "fraudulent": "احتيالية",
        "details_title": "تفاصيل الإدخال المستخدمة في النموذج",
        "history_title": "🗂️ آخر الفحوصات",
        "download_history": "⬇️ تنزيل السجل (CSV)",
        "loading_model": "جاري تحميل النموذج…",
        "model_shape_error": "المتوقع 28 خاصية V، لكن الموجود {n}. أعد التدريب/تصدير الوسيطات.",
        "lang_en": "الإنجليزية",
        "lang_ar": "العربية",
        "lang_es": "الإسبانية",
        "lang_fr": "الفرنسية",
        "lang_ja": "اليابانية",
        "units": {"seconds":"ثوانٍ","minutes":"دقائق","hours":"ساعات","days":"أيام","weeks":"أسابيع","months":"أشهر","years":"سنوات"},
        "currencies": {"USD":"دولار أمريكي","SAR":"ريال سعودي","EUR":"يورو","GBP":"جنيه إسترليني","JPY":"ين ياباني","Custom":"مخصص"},
        "sar_display": "الريال السعودي (SAR)",
        "fx_status_ok": "تم استخدام سعر صرف مباشر",
        "fx_status_fallback": "مشكلة شبكة — تم استخدام أسعار احتياطية"
    },
    "es": {
        "app_title": "Detector de Fraude con Tarjeta",
        "app_tagline": "Ingresa el monto y el tiempo. Elige la moneda y la unidad — convertiremos y evaluaremos la operación. Veredicto claro, confianza y un historial listo para auditorías.",
        "settings": "Configuración",
        "language": "Idioma",
        "hide_dev": "Ocultar menú de desarrollador de Streamlit",
        "threshold": "Umbral de fraude",
        "threshold_help": "Si la probabilidad de fraude ≥ umbral, se marcará como FRAUDE.",
        "currency": "Moneda",
        "fx_label": "Tasa de conversión → {sar_name}",
        "fx_caption": "Usando tasas **en vivo** (se actualizan automáticamente). Puedes sobrescribir si lo necesitas.",
        "fx_override": "Sobrescribir la tasa en vivo para esta moneda",
        "time_unit": "Unidad de tiempo",
        "time_caption": "El tiempo se convierte a horas para coincidir con el entrenamiento (Time_Hours).",
        "amount_input": "Monto de la transacción ({currency_name})",
        "time_input": "Tiempo desde la primera transacción ({unit})",
        "check_btn": "🔎 Verificar transacción",
        "extreme_amount": "⚠️ Este monto es inusualmente alto frente a los datos de entrenamiento (por encima del percentil 99).",
        "evaluating": "Evaluando…",
        "fraud_banner": "🚨 **¡Transacción FRAUDULENTA!** (Confianza: {p:.2%})",
        "legit_banner": "✅ **Transacción legítima.** (Confianza: {p:.2%})",
        "risk_low": "Riesgo bajo",
        "risk_mid": "Riesgo medio",
        "risk_high": "Riesgo alto",
        "risk_card_meta": "Umbral: {thr:.2f} • Prob. de fraude: {p:.2%}",
        "confidence_title": "Confianza de la predicción",
        "legitimate": "Legítima",
        "fraudulent": "Fraudulenta",
        "details_title": "Detalles usados por el modelo",
        "history_title": "🗂️ Revisiones recientes",
        "download_history": "⬇️ Descargar historial (CSV)",
        "loading_model": "Cargando modelo…",
        "model_shape_error": "Se esperaban 28 características V, se obtuvieron {n}.",
        "lang_en": "Inglés","lang_ar":"Árabe","lang_es":"Español","lang_fr":"Francés","lang_ja":"Japonés",
        "units": {"seconds":"segundos","minutes":"minutos","hours":"horas","days":"días","weeks":"semanas","months":"meses","years":"años"},
        "currencies": {"USD":"Dólar estadounidense","SAR":"Riyal saudí","EUR":"Euro","GBP":"Libra esterlina","JPY":"Yen japonés","Custom":"Personalizado"},
        "sar_display": "Riyal saudí (SAR)",
        "fx_status_ok": "Tasa en vivo aplicada",
        "fx_status_fallback": "Sin red — usando tasas de respaldo"
    },
    "fr": {
        "app_title": "Détecteur de Fraude Carte Bancaire",
        "app_tagline": "Saisissez le montant et le temps. Choisissez la devise et l’unité — nous convertirons et noterons l’opération. Verdict clair, confiance et historique prêt pour l’audit.",
        "settings": "Paramètres",
        "language": "Langue",
        "hide_dev": "Masquer le menu développeur Streamlit",
        "threshold": "Seuil de fraude",
        "threshold_help": "Si la probabilité de fraude ≥ seuil, l’opération sera marquée comme FRAUDE.",
        "currency": "Devise",
        "fx_label": "Taux de conversion → {sar_name}",
        "fx_caption": "Taux **en direct** utilisés (mise à jour auto). Vous pouvez forcer un taux si besoin.",
        "fx_override": "Forcer le taux en direct pour cette devise",
        "time_unit": "Unité de temps",
        "time_caption": "Le temps est converti en heures pour correspondre à l’entraînement (Time_Hours).",
        "amount_input": "Montant de l’opération ({currency_name})",
        "time_input": "Temps depuis la première opération ({unit})",
        "check_btn": "🔎 Vérifier l’opération",
        "extreme_amount": "⚠️ Montant inhabituellement élevé (au-dessus du 99e percentile).",
        "evaluating": "Évaluation…",
        "fraud_banner": "🚨 **Opération FRAUDULEUSE détectée !** (Confiance : {p:.2%})",
        "legit_banner": "✅ **Opération légitime.** (Confiance : {p:.2%})",
        "risk_low": "Risque faible",
        "risk_mid": "Risque moyen",
        "risk_high": "Risque élevé",
        "risk_card_meta": "Seuil : {thr:.2f} • Prob. fraude : {p:.2%}",
        "confidence_title": "Confiance de la prédiction",
        "legitimate": "Légitime",
        "fraudulent": "Frauduleuse",
        "details_title": "Détails utilisés par le modèle",
        "history_title": "🗂️ Vérifications récentes",
        "download_history": "⬇️ Télécharger l’historique (CSV)",
        "loading_model": "Chargement du modèle…",
        "model_shape_error": "28 caractéristiques V attendues, {n} obtenues.",
        "lang_en": "Anglais","lang_ar":"Arabe","lang_es":"Espagnol","lang_fr":"Français","lang_ja":"Japonais",
        "units": {"seconds":"secondes","minutes":"minutes","hours":"heures","days":"jours","weeks":"semaines","months":"mois","years":"années"},
        "currencies": {"USD":"Dollar américain","SAR":"Riyal saoudien","EUR":"Euro","GBP":"Livre sterling","JPY":"Yen japonais","Custom":"Personnalisé"},
        "sar_display": "Riyal saoudien (SAR)",
        "fx_status_ok": "Taux en direct appliqué",
        "fx_status_fallback": "Hors-ligne — taux de secours"
    },
    "ja": {
        "app_title": "クレジットカード不正検知",
        "app_tagline": "金額と時間を入力し、通貨と単位を選択してください。自動で換算・判定します。結果・信頼度・履歴を表示します。",
        "settings": "設定",
        "language": "言語",
        "hide_dev": "Streamlit 開発メニューを隠す",
        "threshold": "不正判定しきい値",
        "threshold_help": "不正確率がしきい値以上なら不正としてフラグします。",
        "currency": "通貨",
        "fx_label": "換算レート → {sar_name}",
        "fx_caption": "為替は**自動取得**（定期更新）。必要なら下で上書きできます。",
        "fx_override": "この通貨の自動レートを上書きする",
        "time_unit": "時間の単位",
        "time_caption": "学習と合わせるため、時間は時間（hours）に換算されます。",
        "amount_input": "取引金額（{currency_name}）",
        "time_input": "最初の取引からの経過時間（{unit}）",
        "check_btn": "🔎 取引をチェック",
        "extreme_amount": "⚠️ 学習データと比べて非常に高額です（99パーセンタイル超）。",
        "evaluating": "判定中…",
        "fraud_banner": "🚨 **不正の可能性が高い取引**（信頼度: {p:.2%}）",
        "legit_banner": "✅ **正当な取引**（信頼度: {p:.2%}）",
        "risk_low": "低リスク",
        "risk_mid": "中リスク",
        "risk_high": "高リスク",
        "risk_card_meta": "しきい値: {thr:.2f} • 不正確率: {p:.2%}",
        "confidence_title": "予測の信頼度",
        "legitimate": "正当",
        "fraudulent": "不正",
        "details_title": "モデルが使用した詳細",
        "history_title": "🗂️ 最近のチェック",
        "download_history": "⬇️ 履歴をダウンロード（CSV）",
        "loading_model": "モデル読込中…",
        "model_shape_error": "V特長は28個のはずですが {n} 個でした。再学習/エクスポートしてください。",
        "lang_en": "英語","lang_ar":"アラビア語","lang_es":"スペイン語","lang_fr":"フランス語","lang_ja":"日本語",
        "units": {"seconds":"秒","minutes":"分","hours":"時間","days":"日","weeks":"週","months":"か月","years":"年"},
        "currencies": {"USD":"米ドル","SAR":"サウジリヤル","EUR":"ユーロ","GBP":"英ポンド","JPY":"日本円","Custom":"カスタム"},
        "sar_display": "サウジリヤル（SAR）",
        "fx_status_ok": "為替レートを自動取得",
        "fx_status_fallback": "オフライン — 予備レートを使用"
    },
}
RTL_LANGS = {"ar"}

def t(lang, key, **kwargs):
    val = TRANSLATIONS.get(lang, {}).get(key, TRANSLATIONS["en"].get(key, key))
    if isinstance(val, str):
        return val.format(**kwargs)
    return val

def unit_label(lang, unit_key):
    return TRANSLATIONS.get(lang, {}).get("units", {}).get(unit_key, TRANSLATIONS["en"]["units"][unit_key])

def currency_label(lang, code):
    return TRANSLATIONS.get(lang, {}).get("currencies", {}).get(code, TRANSLATIONS["en"]["currencies"].get(code, code))

# -------------------- Page setup --------------------
st.set_page_config(page_title="Fraud Detector", page_icon="💳", layout="wide")

# -------------------- Language first --------------------
with st.sidebar:
    st.subheader("Settings / الإعدادات")
    lang = st.selectbox(
        "Language / اللغة",
        ["en", "ar", "es", "fr", "ja"],
        index=0,
        format_func=lambda x: TRANSLATIONS[x].get(
            "lang_en" if x=="en" else ("lang_ar" if x=="ar" else ("lang_es" if x=="es" else ("lang_fr" if x=="fr" else "lang_ja"))), x)
    )

# RTL/LTR
st.markdown(
    f"<style>.block-container {{ direction: {'rtl' if lang in RTL_LANGS else 'ltr'}; }}</style>",
    unsafe_allow_html=True,
)

# Dev menu toggle CSS
def set_streamlit_chrome(hide: bool):
    css = """
    <style>
    div[data-testid="stToolbar"] { display: none !important; }
    #MainMenu { visibility: hidden; }
    </style>
    """
    if hide:
        st.markdown(css, unsafe_allow_html=True)

# -------------------- Load artifacts --------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(os.path.join("..", "models", "fraud_model.pkl"))
    scaler = joblib.load(os.path.join("..", "models", "fraud_scaler.pkl"))
    med = joblib.load(os.path.join("..", "models", "fraud_median_features.pkl"))
    amt99 = joblib.load(os.path.join("..", "models", "amount_99th_percentile.pkl"))
    return model, scaler, med, amt99

with st.spinner(t(lang, "loading_model")):
    model, scaler, median_features, amount_99 = load_artifacts()

if len(median_features) != 28:
    st.error(t(lang, "model_shape_error", n=len(median_features))); st.stop()

# -------------------- Styles --------------------
st.markdown("""
<style>
.block-container {padding-top: 2.0rem; padding-bottom: 2rem; max-width: 1100px;}
.card {border-radius: 16px; padding: 1.1rem 1.2rem; border: 1px solid rgba(255,255,255,0.08); background: rgba(0,0,0,0.25);}
.metric-badge {display:inline-block; padding: .35rem .7rem; border-radius: 999px; font-weight:600;}
.badge-low {background: rgba(25,135,84,.15); color: #52d29a; border: 1px solid rgba(25,135,84,.35);}
.badge-med {background: rgba(255,193,7,.12); color: #ffcf66; border: 1px solid rgba(255,193,7,.35);}
.badge-high{background: rgba(220,53,69,.15); color: #ff7b89; border: 1px solid rgba(220,53,69,.35);}
.small {opacity:.8; font-size:.88rem;}
hr {border: 0; height: 1px; background: linear-gradient(to right, transparent, rgba(255,255,255,.18), transparent);}
</style>
""", unsafe_allow_html=True)

# -------------------- FX: live rates to SAR --------------------
FALLBACK_RATES_TO_SAR = {  # 1 unit of code -> SAR
    "USD": 3.75, "SAR": 1.0, "EUR": 4.05, "GBP": 4.75, "JPY": 0.026
}

@st.cache_data(show_spinner=False, ttl=60*60*6)  # refresh every 6 hours
def fetch_fx_to_sar():
    # Use exchangerate.host free API (no key). We get rates with base=SAR,
    # then invert to get 1 unit of currency -> SAR.
    url = "https://api.exchangerate.host/latest?base=SAR&symbols=USD,EUR,GBP,JPY"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        sar_base = data.get("rates", {})
        # sar_base maps currency -> amount per 1 SAR (e.g., USD per SAR)
        to_sar = {"SAR": 1.0}
        for code, per_sar in sar_base.items():
            if per_sar and per_sar > 0:
                to_sar[code] = 1.0 / per_sar  # 1 unit of code -> SAR
        # Ensure all needed codes exist, fill with fallbacks if missing
        for c, v in FALLBACK_RATES_TO_SAR.items():
            to_sar.setdefault(c, v)
        status = "ok"
    except Exception:
        to_sar = FALLBACK_RATES_TO_SAR.copy()
        status = "fallback"
    return to_sar, status

fx_map_to_sar, fx_status = fetch_fx_to_sar()

# -------------------- Helpers --------------------
def risk_tier_from_model(p, thr):
    if p >= max(thr, 0.66): return 2
    if p >= 0.33: return 1
    return 0

def risk_tier_from_rules(amount_sar, time_hours, amount_99):
    usd = amount_sar / FALLBACK_RATES_TO_SAR["USD"]  # approximate USD for rules
    if amount_sar >= amount_99 or usd >= 8000 or (usd >= 2000 and time_hours <= 2):
        return 2
    if (500 <= usd < 2000 and time_hours <= 6) or (2000 <= usd < 8000 and time_hours <= 12):
        return 1
    return 0

def tier_label_badge(lang, tier):
    if tier == 2: return t(lang, "risk_high"), "badge-high"
    if tier == 1: return t(lang, "risk_mid"),  "badge-med"
    return t(lang, "risk_low"), "badge-low"

# -------------------- Sidebar --------------------
with st.sidebar:
    st.subheader(t(lang, "settings"))

    hide_dev = st.checkbox(t(lang, "hide_dev"), value=False)
    set_streamlit_chrome(hide_dev)

    threshold = st.slider(t(lang, "threshold"), 0.30, 0.90, 0.60, 0.01, help=t(lang, "threshold_help"))

    st.markdown(f"### 💱 {t(lang, 'currency')}")
    currency_codes = ["USD", "SAR", "EUR", "GBP", "JPY", "Custom"]
    currency = st.selectbox(
        " ",
        currency_codes,
        format_func=lambda c: f"{c} — {currency_label(lang, c)}",
        label_visibility="collapsed"
    )

    # Determine live FX to SAR
    live_rate = fx_map_to_sar.get(currency, 1.0)
    sar_name = t(lang, "sar_display")

    st.caption(t(lang, "fx_caption"))
    st.info((t(lang, "fx_status_ok") if fx_status=="ok" else t(lang, "fx_status_fallback")) + f" • 1 {currency} ≈ {live_rate:.6f} SAR")

    fx_override_on = False
    if currency != "Custom":
        fx_override_on = st.checkbox(t(lang, "fx_override"), value=False)
    # If Custom OR override toggled, allow entering a rate
    if currency == "Custom" or fx_override_on:
        fx = st.number_input(t(lang, "fx_label", sar_name=sar_name), min_value=0.0, value=float(live_rate if currency!="Custom" else 3.75), step=0.0001, format="%.6f")
    else:
        fx = float(live_rate)

    st.markdown(f"### ⏱️ {t(lang, 'time_unit')}")
    unit_keys = ["seconds", "minutes", "hours", "days", "weeks", "months", "years"]
    time_unit_key = st.selectbox("  ", unit_keys, format_func=lambda k: unit_label(lang, k), label_visibility="collapsed")
    st.caption(t(lang, "time_caption"))

# -------------------- Header --------------------
st.title(f"💳 {t(lang, 'app_title')}")
st.write(t(lang, "app_tagline"))
st.markdown("<hr/>", unsafe_allow_html=True)

# -------------------- Inputs --------------------
col1, col2, col3 = st.columns([1.1, 1.1, 1])
with col1:
    amount_label = t(lang, "amount_input", currency_name=currency_label(lang, currency))
    amount_input = st.number_input(amount_label, min_value=0.0, value=0.0, step=1.0, format="%.2f")
with col2:
    time_label = t(lang, "time_input", unit=unit_label(lang, time_unit_key))
    time_value = st.number_input(time_label, min_value=0, value=0, step=1)
with col3:
    st.write(""); st.write("")
    run = st.button(t(lang, "check_btn"), use_container_width=True)

# -------------------- Conversions --------------------
to_hours = {
    "seconds": 1/3600,
    "minutes": 1/60,
    "hours": 1,
    "days": 24,
    "weeks": 24*7,
    "months": 24*30.44,
    "years": 24*365.25,
}
amount_sar = float(amount_input * fx)
time_hours = float(int(time_value) * to_hours[time_unit_key])

if amount_sar > amount_99:
    st.warning(t(lang, "extreme_amount"))

# -------------------- Prediction --------------------
if 'history' not in st.session_state:
    st.session_state.history = []

@st.cache_data(show_spinner=False, ttl=0)
def predict_once_cached(vec_arr):
    scaled = scaler.transform(vec_arr)
    return float(model.predict_proba(scaled)[0][1])

def predict_once():
    vec = np.array([*median_features, time_hours, amount_sar], dtype=float).reshape(1, -1)
    proba_fraud = predict_once_cached(vec)
    tier_model = risk_tier_from_model(proba_fraud, threshold)
    tier_rules = risk_tier_from_rules(amount_sar, time_hours, amount_99)
    final_tier = max(tier_model, tier_rules)
    is_fraud = (proba_fraud >= threshold)
    return proba_fraud, final_tier, is_fraud

if run:
    with st.spinner(t(lang, "evaluating")):
        time.sleep(0.15)
        proba_fraud, final_tier, is_fraud = predict_once()

    if is_fraud:
        st.error(t(lang, "fraud_banner", p=proba_fraud))
    else:
        st.success(t(lang, "legit_banner", p=(1 - proba_fraud)))

    risk_text, badge = tier_label_badge(lang, final_tier)
    st.markdown(f"""
    <div class="card" style="margin-top:.75rem;">
        <span class="metric-badge {badge}">{risk_text}</span>
        <div class="small">{t(lang, "risk_card_meta", thr=threshold, p=proba_fraud)}</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bar
    labels = [t(lang, "legitimate"), t(lang, "fraudulent")]
    scores = [1 - proba_fraud, proba_fraud]
    colors = ['green', 'red']
    fig, ax = plt.subplots(figsize=(6.5, 1.8))
    bars = ax.barh(labels, scores, color=colors)
    ax.set_xlim(0, 1)
    ax.bar_label(bars, fmt='%.2f', label_type='center', color='white')
    ax.set_title(t(lang, "confidence_title"))
    st.pyplot(fig)

    with st.expander(t(lang, "details_title")):
        st.write({
            "Amount_SAR": float(amount_sar),
            "Time_Hours": float(time_hours),
            "Input Amount": float(amount_input),
            "Currency": f"{currency} — {currency_label(lang, currency)}",
            "FX → SAR": float(fx),
            "FX_Source": "live" if (not fx_override_on and currency!="Custom" and fx_status=="ok") else ("fallback" if fx_status!="ok" and not fx_override_on else "override"),
            "Time Value (int)": int(time_value),
            "Time Unit": unit_label(lang, time_unit_key)
        })

    st.session_state.history.append({
        "Amount": float(amount_input),
        "Currency": currency,
        "Currency_Name": currency_label(lang, currency),
        "FX→SAR": float(fx),
        "Time": int(time_value),
        "Unit": time_unit_key,
        "Amount_SAR": float(amount_sar),
        "Time_Hours": float(time_hours),
        "Fraud_Prob": float(proba_fraud),
        "Threshold": float(threshold),
        "RiskTier": ["Low","Mid","High"][final_tier],
        "Verdict": "FRAUD" if is_fraud else "LEGIT"
    })

# -------------------- History --------------------
if st.session_state.history:
    st.markdown(f"### {t(lang, 'history_title')}")
    dfh = pd.DataFrame(st.session_state.history[::-1])
    st.dataframe(dfh, use_container_width=True, height=260)
    st.download_button(
        t(lang, "download_history"),
        data=dfh.to_csv(index=False).encode("utf-8"),
        file_name="fraud_checks_history.csv",
        mime="text/csv",
        use_container_width=True
    )
