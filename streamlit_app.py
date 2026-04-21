import streamlit as st
import pickle
import json
import numpy as np
import os

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}

/* Hide default Streamlit header */
#MainMenu, footer, header {visibility: hidden;}

/* Hero section */
.hero-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}

.hero-subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 2rem;
    font-weight: 400;
}

/* Card container */
.card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 25px 50px rgba(0,0,0,0.4);
    margin-bottom: 1.5rem;
}

/* Section labels */
.section-label {
    color: #a78bfa;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, rgba(167,139,250,0.15), rgba(96,165,250,0.15));
    border: 1px solid rgba(167,139,250,0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: fadeInUp 0.5s ease;
}

.result-amount {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.result-label {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 0.4rem;
    font-weight: 500;
}

/* Risk badge */
.badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    margin-top: 0.8rem;
}
.badge-low    { background: rgba(52,211,153,0.2); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
.badge-medium { background: rgba(251,191,36,0.2); color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); }
.badge-high   { background: rgba(248,113,113,0.2); color: #f87171; border: 1px solid rgba(248,113,113,0.4); }

/* Streamlit widget styling overrides */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: rgba(255,255,255,0.07) !important;
    border-color: rgba(255,255,255,0.18) !important;
    border-radius: 10px !important;
    color: white !important;
}

.stSlider > div > div > div { background: #a78bfa !important; }

label, .stSelectbox label, .stSlider label, .stNumberInput label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
}

/* Button */
div.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    width: 100%;
    transition: all 0.3s ease !important;
    letter-spacing: 0.03em;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
}

div.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.6) !important;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Info boxes */
.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
}
.info-item {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
}
.info-value { font-size: 1.3rem; font-weight: 700; color: #a78bfa; }
.info-key   { font-size: 0.75rem; color: #94a3b8; margin-top: 0.2rem; }

hr.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.1);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNN_PKL   = os.path.join(BASE_DIR, "Insurance_KNN_Project", "KNN_Model_Insurance.pkl")
JSON_PATH = os.path.join(BASE_DIR, "Insurance_KNN_Project", "knn_encode_json.json")
SCALER    = os.path.join(BASE_DIR, "Insurance_KNN_Project", "Std_Scaler.pkl")

@st.cache_resource
def load_models():
    with open(KNN_PKL,   "rb") as f: knn_model  = pickle.load(f)
    with open(JSON_PATH, "r")  as f: encode_map = json.load(f)
    with open(SCALER,    "rb") as f: scaler     = pickle.load(f)
    return knn_model, encode_map, scaler


def predict_charges(age, bmi, children, smoker, region, knn_model, encode_map, scaler):
    test_array = np.zeros(len(encode_map["columns"]))
    test_array[0] = children
    test_array[1] = encode_map["smoker"][smoker.lower()]
    region_key = "region_" + region.lower()
    region_idx = encode_map["columns"].index(region_key)
    test_array[region_idx] = 1
    # age bucket
    test_array[6] = 0 if age < 18 else 1 if age < 30 else 2 if age < 45 else 3 if age < 60 else 4
    test_array[7] = 1 if age > 60 else 0      # risk
    test_array[8] = 1 if bmi > 25 else 0      # weight
    std_arr = scaler.transform([test_array])
    return np.around(knn_model.predict(std_arr)[0], 2)


def risk_badge(charges):
    if charges < 8000:
        return '<span class="badge badge-low">🟢 LOW RISK</span>'
    elif charges < 20000:
        return '<span class="badge badge-medium">🟡 MEDIUM RISK</span>'
    else:
        return '<span class="badge badge-high">🔴 HIGH RISK</span>'


# ─── App ────────────────────────────────────────────────────────────────────────
def main():
    # Hero
    st.markdown('<h1 class="hero-title">🏥 Medical Insurance Cost Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Powered by K-Nearest Neighbours · Instant AI-based charge estimation</p>', unsafe_allow_html=True)

    try:
        knn_model, encode_map, scaler = load_models()
    except FileNotFoundError as e:
        st.error(f"❌ Model file not found: {e}")
        st.info("Make sure the `Insurance_KNN_Project/` folder contains `KNN_Model_Insurance.pkl`, `Std_Scaler.pkl`, and `knn_encode_json.json`.")
        return

    # ── Input Card ──────────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">📋 Patient Information</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=100, value=30, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5, step=0.1, format="%.1f")
    with col2:
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
        smoker   = st.selectbox("Smoker?", ["No", "Yes"])

    region = st.selectbox(
        "Region",
        ["Northeast", "Northwest", "Southeast", "Southwest"],
        help="Geographic region of the insured person"
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Summary Card ─────────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="section-label">📊 Input Summary</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-grid">
        <div class="info-item"><div class="info-value">{age}</div><div class="info-key">Age</div></div>
        <div class="info-item"><div class="info-value">{bmi:.1f}</div><div class="info-key">BMI</div></div>
        <div class="info-item"><div class="info-value">{children}</div><div class="info-key">Children</div></div>
        <div class="info-item"><div class="info-value">{"🚬 Yes" if smoker=="Yes" else "✅ No"}</div><div class="info-key">Smoker</div></div>
        <div class="info-item"><div class="info-value">{region}</div><div class="info-key">Region</div></div>
        <div class="info-item"><div class="info-value">{"Overweight" if bmi>25 else "Normal"}</div><div class="info-key">Weight Status</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict Button ────────────────────────────────────────────────────────────
    if st.button("⚡ Predict Insurance Charges"):
        with st.spinner("Running KNN model..."):
            try:
                charges = predict_charges(
                    age, bmi, children, smoker, region,
                    knn_model, encode_map, scaler
                )
                badge = risk_badge(charges)
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">Estimated Annual Insurance Charges</div>
                    <div class="result-amount">${charges:,.2f}</div>
                    {badge}
                    <div class="result-label" style="margin-top:1rem;">
                        Based on KNN regression model trained on US medical insurance data
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Extra tips
                st.markdown('<hr class="divider">', unsafe_allow_html=True)
                st.markdown("**💡 Key Factors Influencing Your Estimate:**")
                tips = []
                if smoker == "Yes":
                    tips.append("🚬 **Smoking** is the single largest cost driver — quitting could reduce charges by 3-4×.")
                if bmi > 25:
                    tips.append(f"⚖️ **BMI {bmi:.1f}** (overweight) — maintaining a healthy BMI (<25) can lower premiums.")
                if age > 45:
                    tips.append(f"👴 **Age {age}** — older age groups typically pay higher premiums.")
                if children >= 3:
                    tips.append(f"👨‍👩‍👧‍👦 **{children} dependents** — more children can increase your premium.")
                if not tips:
                    tips.append("✅ Your profile shows several low-risk factors — great news for your premium!")
                for tip in tips:
                    st.markdown(f"- {tip}")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")

    # ── Footer ────────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#475569;font-size:0.8rem;">'
        'Medical Insurance Cost Predictor · KNN Model · For informational purposes only'
        '</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
