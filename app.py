import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ────────────────────────────────────────────────────────────────────────────
# PART 1: MODEL TRAINING
# ────────────────────────────────────────────────────────────────────────────

def train_model(csv_path):
    """Train the customer churn prediction model"""
    try:
        # Load data
        df = pd.read_csv(csv_path)
        print(f"Data shape: {df.shape}")
        print(df.head(2))
        
        # Data cleaning
        df = df.drop(columns=["customerID"])
        
        # Convert TotalCharges to float
        df["TotalCharges"] = df["TotalCharges"].replace({" ": "0.0"})
        df["TotalCharges"] = df["TotalCharges"].astype(float)
        
        # Encode target
        df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0})
        
        # Get object columns
        object_columns = df.select_dtypes(include="object").columns
        
        # Initialize encoders dictionary
        encoders = {}
        for column in object_columns:
            label_encoder = LabelEncoder()
            df[column] = label_encoder.fit_transform(df[column])
            encoders[column] = label_encoder
        
        # Save encoders
        with open("encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)
        print("✅ Encoders saved to encoders.pkl")
        
        # Split features and target
        X = df.drop(columns=["Churn"])
        y = df["Churn"]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"Training set: {y_train.value_counts().to_dict()}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {y_train_smote.value_counts().to_dict()}")
        
        # Train multiple models
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, verbosity=0),
        }
        
        cv_scores = {}
        for model_name, model in models.items():
            print(f"\n🔄 Training {model_name}...")
            scores = cross_val_score(
                model, X_train_smote, y_train_smote, cv=5, scoring="accuracy"
            )
            cv_scores[model_name] = scores
            print(f"{model_name} CV Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        
        # Train Random Forest (best model)
        print("\n🚀 Training final Random Forest model...")
        rfc = RandomForestClassifier(random_state=42)
        rfc.fit(X_train_smote, y_train_smote)
        
        # Evaluate
        y_pred = rfc.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Test Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        model_data = {"model": rfc, "features_names": X.columns.tolist()}
        with open("customer_churn_model.pkl", "wb") as f:
            pickle.dump(model_data, f)
        print("\n✅ Model saved to customer_churn_model.pkl")
        
        return rfc, encoders, X.columns.tolist(), accuracy
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        raise

# ────────────────────────────────────────────────────────────────────────────
# PART 2: STREAMLIT UI
# ────────────────────────────────────────────────────────────────────────────

# Logo SVG (works perfectly in Streamlit)
LOGO_SVG = """
<svg width="130" height="130" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <!-- Background -->
  <defs>
    <linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#38bdf8;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#818cf8;stop-opacity:1" />
    </linearGradient>
  </defs>
  
  <!-- Circle background -->
  <circle cx="100" cy="100" r="95" fill="url(#logoGrad)" opacity="0.1" stroke="#38bdf8" stroke-width="2"/>
  
  <!-- Signal waves -->
  <g fill="none" stroke="#38bdf8" stroke-width="3" stroke-linecap="round">
    <path d="M 100 160 Q 70 130 70 100 Q 70 70 100 40" opacity="0.3"/>
    <path d="M 100 150 Q 80 125 80 100 Q 80 75 100 50" opacity="0.6"/>
    <path d="M 100 140 Q 90 120 90 100 Q 90 80 100 60" opacity="1"/>
  </g>
  
  <!-- Center dot -->
  <circle cx="100" cy="100" r="8" fill="#38bdf8"/>
  
  <!-- Text -->
  <text x="100" y="180" font-family="Arial, sans-serif" font-size="14" font-weight="bold" 
        text-anchor="middle" fill="#38bdf8">CHURN AI</text>
</svg>
"""

def logo_html(width=130):
    """Return SVG logo as HTML"""
    svg = LOGO_SVG.replace('width="130"', f'width="{width}"').replace('height="130"', f'height="{width}"')
    return f'<div style="display:flex;justify-content:center;align-items:center;">{svg}</div>'

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction | Claysys Technologies",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root{
  --bg:#07090f; --surf:#0e1420; --surf2:#141d2e;
  --border:rgba(56,189,248,.14); --accent:#38bdf8; --accent2:#818cf8;
  --churn:#f87171; --safe:#34d399; --text:#e2e8f0; --muted:#64748b;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:var(--bg);color:var(--text);}
.stApp{background:radial-gradient(ellipse at 18% 0%,#0d1f3c 0%,#07090f 58%);}
#MainMenu,footer,header{visibility:hidden;}

.topbar{
  display:flex;align-items:center;justify-content:space-between;
  padding:12px 28px;background:rgba(14,20,32,.96);
  border-bottom:1px solid var(--border);border-radius:0 0 16px 16px;
  margin-bottom:28px;backdrop-filter:blur(10px);position:sticky;top:0;z-index:999;
}
.topbar-r{font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--muted);}
.live-tag{background:rgba(56,189,248,.1);border:1px solid rgba(56,189,248,.3);color:var(--accent);font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;padding:3px 12px;border-radius:20px;margin-left:10px;}

.hero{
  display:grid;grid-template-columns:1fr auto;gap:24px;align-items:center;
  background:linear-gradient(130deg,#0e1420,#141d2e);border:1px solid var(--border);border-radius:20px;
  padding:44px 52px;margin-bottom:30px;position:relative;overflow:hidden;
}
.hero::before{content:'';position:absolute;top:-80px;right:8%;width:340px;height:340px;background:radial-gradient(circle,rgba(56,189,248,.1) 0%,transparent 70%);pointer-events:none;}
.eye{font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:var(--accent);margin-bottom:14px;}
.hero h1{font-family:'Syne',sans-serif;font-size:44px;font-weight:800;line-height:1.08;margin:0 0 14px;background:linear-gradient(100deg,#fff 30%,#38bdf8 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hdesc{color:var(--muted);font-size:15px;font-weight:300;max-width:520px;}
.hstats{display:flex;gap:28px;margin-top:28px;}
.hsv{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:var(--accent);line-height:1;}
.hsl{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-top:4px;}
.hlogo{background:rgba(255,255,255,.04);border:1px solid var(--border);border-radius:16px;padding:22px 30px;display:flex;align-items:center;justify-content:center;min-width:160px;width:180px;height:180px;}

.shdr{display:flex;align-items:center;gap:10px;font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:2.5px;text-transform:uppercase;color:var(--accent);margin:32px 0 18px;}
.shdr::after{content:'';flex:1;height:1px;background:var(--border);}

.icard{background:var(--surf);border:1px solid var(--border);border-radius:16px;padding:22px 22px 8px;}
.icard-t{font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--muted);margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid var(--border);}

.stButton>button{
  background:linear-gradient(135deg,#38bdf8,#818cf8) !important;
  color:#07090f !important;border:none !important;border-radius:14px !important;
  font-family:'Syne',sans-serif !important;font-weight:800 !important;
  font-size:15px !important;letter-spacing:1px !important;text-transform:uppercase !important;
  padding:16px 32px !important;width:100% !important;
  box-shadow:0 6px 28px rgba(56,189,248,.30) !important;transition:all .3s !important;
}
.stButton>button:hover{transform:translateY(-3px) !important;box-shadow:0 14px 44px rgba(56,189,248,.45) !important;}

.res-c{background:linear-gradient(135deg,rgba(248,113,113,.13),rgba(248,113,113,.04));border:1px solid rgba(248,113,113,.45);border-radius:20px;padding:36px;text-align:center;margin:24px 0;}
.res-s{background:linear-gradient(135deg,rgba(52,211,153,.13),rgba(52,211,153,.04));border:1px solid rgba(52,211,153,.45);border-radius:20px;padding:36px;text-align:center;margin:24px 0;}
.ricon{font-size:60px;margin-bottom:10px;}
.rlbl{font-family:'Syne',sans-serif;font-size:30px;font-weight:800;margin:0;}
.rsub{color:var(--muted);font-size:14px;margin-top:8px;}

.pb{margin:18px 0;}
.pbm{display:flex;justify-content:space-between;font-size:13px;color:var(--muted);margin-bottom:7px;}
.pbbg{background:rgba(255,255,255,.06);border-radius:99px;height:10px;overflow:hidden;}
.pbc{height:100%;border-radius:99px;background:linear-gradient(90deg,#f87171,#fca5a5);}
.pbs{height:100%;border-radius:99px;background:linear-gradient(90deg,#34d399,#6ee7b7);}

.tiles{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0;}
.tile{background:var(--surf2);border:1px solid var(--border);border-radius:12px;padding:14px 18px;flex:1;min-width:100px;text-align:center;}
.tv{font-family:'Syne',sans-serif;font-size:22px;font-weight:700;color:var(--accent);}
.tl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1.2px;margin-top:3px;}

.rbadge{border-radius:12px;padding:18px;text-align:center;margin-top:10px;}
.rbl{font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;}
.rbv{font-family:'Syne',sans-serif;font-size:26px;font-weight:800;margin-top:4px;}

.rcard{background:var(--surf);border:1px solid var(--border);border-radius:14px;padding:22px;}
.rico{font-size:30px;margin-bottom:8px;}
.rtit{font-family:'Syne',sans-serif;font-weight:700;font-size:14px;margin-bottom:6px;}
.rdsc{font-size:13px;color:var(--muted);line-height:1.7;}

.footer{text-align:center;padding:28px 0 12px;border-top:1px solid var(--border);margin-top:40px;}
.fcpy{font-size:11px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin-top:10px;}

section[data-testid="stSidebar"]{background:var(--surf) !important;border-right:1px solid var(--border) !important;}
section[data-testid="stSidebar"] *{color:var(--text) !important;}
.sbsub{font-size:11px;color:var(--muted);margin-bottom:20px;letter-spacing:1px;text-transform:uppercase;}
label,div[data-testid="stWidgetLabel"]{color:#94a3b8 !important;font-size:12px !important;font-weight:500 !important;}
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load trained model and encoders"""
    model_path = "customer_churn_model.pkl"
    encoder_path = "encoders.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        return None, None, None, (
            "⚠️ Model files not found. Training model now... "
            "Please wait a moment."
        )
    
    try:
        with open(model_path, "rb") as f:
            md = pickle.load(f)
        with open(encoder_path, "rb") as f:
            enc = pickle.load(f)
        return md["model"], md["features_names"], enc, None
    except Exception as e:
        return None, None, None, f"❌ Error loading model: {e}"

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(logo_html(100), unsafe_allow_html=True)
    st.markdown('<div class="sbsub">AI Analytics Division</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🗺️ How It Works")
    st.markdown("""<div style="font-size:13px;color:#64748b;line-height:2.1;">
    1️⃣ &nbsp;Fill in the customer details<br>
    2️⃣ &nbsp;Click <strong style="color:#38bdf8">Predict Churn Risk</strong><br>
    3️⃣ &nbsp;View churn probability score<br>
    4️⃣ &nbsp;Act on the retention tips
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🤖 Model Details")
    st.markdown("""<div style="font-size:13px;color:#64748b;line-height:2.1;">
    🌲 Random Forest Classifier<br>
    ⚖️ SMOTE class balancing<br>
    📊 5-fold cross-validation<br>
    📁 IBM Telco Churn Dataset<br>
    🎯 ~80%+ test accuracy
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("© 2026 Claysys Technologies · All rights reserved.")

# ── Topbar ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 5])
with col1:
    st.markdown(logo_html(80), unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div style="padding-top: 20px;">
        <div class="topbar-r">
            Customer Churn Prediction
            <span class="live-tag">Live · ML Powered</span>
        </div>
    </div>""", unsafe_allow_html=True)

st.divider()

# ── Hero ───────────────────────────────────────────────────────────────────
c_text, c_logo = st.columns([2, 1])

with c_text:
    st.markdown(f"""
    <div class="eye">🔬 Predictive Analytics · Telecom AI</div>
    <h1 style="font-family:'Syne',sans-serif;font-size:44px;font-weight:800;line-height:1.08;margin:0 0 14px;background:linear-gradient(100deg,#fff 30%,#38bdf8 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">
        Customer Churn<br>Intelligence Platform
    </h1>
    <div class="hdesc">
        Harness machine learning to identify at-risk customers before they leave —
        and act in real time to retain them.
    </div>
    <div class="hstats">
        <div><div class="hsv">~80%</div><div class="hsl">Accuracy</div></div>
        <div><div class="hsv">19</div><div class="hsl">Features</div></div>
        <div><div class="hsv">7K+</div><div class="hsl">Records</div></div>
    </div>""", unsafe_allow_html=True)

with c_logo:
    st.markdown(logo_html(150), unsafe_allow_html=True)

st.divider()

# ── Main Loading ───────────────────────────────────────────────────────────
with st.spinner("⚙️ Loading prediction model..."):
    _model, _features, _encoders, _err = load_model()
    
    # If model not found, train it
    if _err and "not found" in _err:
        try:
            csv_file = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
            if os.path.exists(csv_file):
                st.info("🔄 Training model...")
                train_model(csv_file)
                _model, _features, _encoders, _err = load_model()
        except Exception as e:
            st.error(f"❌ Could not train model: {e}")
            st.stop()

if _err:
    st.error(_err)
    st.stop()

st.success("✅ Model loaded successfully — fill in customer details below.")

# ── Input Form ─────────────────────────────────────────────────────────────
st.markdown('<div class="shdr">Customer Profile Input</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.markdown('<div class="icard"><div class="icard-t">👤 Demographics</div>', unsafe_allow_html=True)
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="icard"><div class="icard-t">📱 Services</div>', unsafe_allow_html=True)
    phone_svc = st.selectbox("Phone Service", ["No", "Yes"])
    multi_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_sup = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    stream_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    stream_mov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="icard"><div class="icard-t">💳 Billing & Contract</div>', unsafe_allow_html=True)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_chg = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    total_chg = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0,
                                value=float(round(monthly_chg * tenure, 2)), step=1.0)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("🔍 Predict Churn Risk", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────
if predict_btn:
    row = {
        'gender': gender,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_svc,
        'MultipleLines': multi_lines,
        'InternetService': internet,
        'OnlineSecurity': online_sec,
        'OnlineBackup': backup,
        'DeviceProtection': device_prot,
        'TechSupport': tech_sup,
        'StreamingTV': stream_tv,
        'StreamingMovies': stream_mov,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_chg,
        'TotalCharges': total_chg,
    }
    inp_df = pd.DataFrame([row])
    
    try:
        for col, enc in _encoders.items():
            if col in inp_df.columns:
                inp_df[col] = enc.transform(inp_df[col])
    except Exception as e:
        st.error(f"❌ Encoding error: {e}")
        st.stop()
    
    for col in _features:
        if col not in inp_df.columns:
            inp_df[col] = 0
    inp_df = inp_df[_features]
    
    pred = _model.predict(inp_df)[0]
    proba = _model.predict_proba(inp_df)[0]
    churn_p = proba[1] * 100
    safe_p = proba[0] * 100
    
    if churn_p >= 70:
        rc, rl, rb = "#f87171", "CRITICAL", "rgba(248,113,113,0.10)"
    elif churn_p >= 40:
        rc, rl, rb = "#fbbf24", "MODERATE", "rgba(251,191,36,0.10)"
    else:
        rc, rl, rb = "#34d399", "LOW", "rgba(52,211,153,0.10)"
    
    st.markdown('<div class="shdr">Prediction Result</div>', unsafe_allow_html=True)
    r1, r2 = st.columns([1.3, 1], gap="medium")
    
    with r1:
        if pred == 1:
            st.markdown("""<div class="res-c">
                <div class="ricon">⚠️</div>
                <div class="rlbl" style="color:#f87171;">HIGH CHURN RISK</div>
                <div class="rsub">This customer is likely to churn. Immediate action recommended.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="res-s">
                <div class="ricon">✅</div>
                <div class="rlbl" style="color:#34d399;">LOW CHURN RISK</div>
                <div class="rsub">This customer is likely to stay. Keep delivering great service!</div>
            </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="pb">
            <div class="pbm"><span>🔴 Churn Probability</span><span><b>{churn_p:.1f}%</b></span></div>
            <div class="pbbg"><div class="pbc" style="width:{churn_p}%"></div></div>
        </div>
        <div class="pb">
            <div class="pbm"><span>🟢 Retention Probability</span><span><b>{safe_p:.1f}%</b></span></div>
            <div class="pbbg"><div class="pbs" style="width:{safe_p}%"></div></div>
        </div>""", unsafe_allow_html=True)
    
    with r2:
        st.markdown(f"""
        <div class="tiles">
            <div class="tile"><div class="tv">{tenure}</div><div class="tl">Months</div></div>
            <div class="tile"><div class="tv">${monthly_chg:.0f}</div><div class="tl">Monthly</div></div>
            <div class="tile"><div class="tv">${total_chg:.0f}</div><div class="tl">Total</div></div>
        </div>
        <div class="tiles">
            <div class="tile"><div class="tv" style="font-size:13px;">{contract}</div><div class="tl">Contract</div></div>
            <div class="tile"><div class="tv" style="font-size:13px;">{internet}</div><div class="tl">Internet</div></div>
        </div>
        <div class="rbadge" style="background:{rb};border:1px solid {rc}44;">
            <div class="rbl">Overall Risk Level</div>
            <div class="rbv" style="color:{rc};">{rl}</div>
        </div>""", unsafe_allow_html=True)
    
    if pred == 1:
        st.markdown('<div class="shdr">Retention Recommendations</div>', unsafe_allow_html=True)
        RECS = [
            ("💰", "Loyalty Discount",
             "Offer a personalised discount on monthly charges to incentivise contract renewal."),
            ("📞", "Proactive Outreach",
             "Schedule a call with a dedicated account manager to resolve pain points early."),
            ("🎁", "Complimentary Upgrade",
             "Offer a free upgrade — Tech Support, Device Protection, or extra streaming bundle."),
        ]
        ca, cb, cc = st.columns(3, gap="medium")
        for col, (ico, tit, dsc) in zip([ca, cb, cc], RECS):
            with col:
                st.markdown(f"""<div class="rcard">
                    <div class="rico">{ico}</div>
                    <div class="rtit">{tit}</div>
                    <div class="rdsc">{dsc}</div>
                </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown(f"""
<div class="footer">
    {logo_html(90)}
    <div class="fcpy">
        Customer Churn Prediction &nbsp;·&nbsp; Claysys Technologies &nbsp;·&nbsp;
        AI &amp; Machine Learning Division &nbsp;·&nbsp; © 2026
    </div>
</div>""", unsafe_allow_html=True)