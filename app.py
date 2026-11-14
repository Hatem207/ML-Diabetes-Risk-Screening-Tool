
import streamlit as st
import pandas as pd
import joblib

# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Diabetes Risk Screening",
    page_icon="🩺",
    layout="centered"
)

# =========================
# CUSTOM CSS DESIGN
# =========================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #eef7f8 0%, #f4f7fc 55%, #edf3fa 100%);
}

/* Main content width */
.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */
.hero {
    background: linear-gradient(120deg, #087f8c, #145b83);
    padding: 35px 30px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 8px 25px rgba(8, 127, 140, 0.18);
}

.hero h1 {
    color: white;
    font-size: 32px;
    margin-bottom: 10px;
    font-weight: 750;
}

.hero p {
    color: #e1f5f5;
    font-size: 15px;
    margin-bottom: 0;
}

/* Section headings */
.section-title {
    color: #126477;
    font-size: 21px;
    font-weight: 700;
    margin-top: 12px;
    margin-bottom: 5px;
}

/* Form card */
div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.92);
    padding: 25px;
    border: 1px solid #dcebed;
    border-radius: 18px;
    box-shadow: 0 5px 20px rgba(35, 75, 95, 0.06);
}

/* Input labels */
label {
    color: #244b5a !important;
    font-weight: 550 !important;
}

/* Input fields */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px;
    border-color: #d5e5e9;
}

/* Submit button */
div.stFormSubmitButton > button {
    width: 100%;
    background: linear-gradient(100deg, #078b91, #126b91);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 13px;
    font-size: 17px;
    font-weight: 700;
    transition: 0.2s;
}

div.stFormSubmitButton > button:hover {
    background: linear-gradient(100deg, #066e78, #104f78);
    color: white;
    border: none;
}

/* Result cards */
.result-card {
    background: white;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #dcebed;
    box-shadow: 0 5px 20px rgba(35, 75, 95, 0.07);
    margin-top: 15px;
    text-align: center;
}

.result-score {
    font-size: 42px;
    font-weight: 800;
    color: #087f8c;
    margin: 8px 0;
}

.result-label {
    color: #55717e;
    font-size: 14px;
}

/* Footer */
.footer {
    text-align: center;
    color: #718995;
    font-size: 12px;
    margin-top: 35px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_bundle():
    return joblib.load("diabetes_risk_model.pkl")


bundle = load_bundle()

model = bundle["model"]
threshold = bundle["threshold"]
features = bundle["features"]
model_name = bundle["model_name"]


# =========================
# APP HEADER
# =========================

st.markdown("""
<div class="hero">
    <div style="font-size:45px;">🩺</div>
    <h1>Diabetes Risk Screening</h1>
    <p>
        Understand your health indicators through
        a machine learning screening tool.
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "This tool is designed for educational screening support only. "
    "It is not a medical diagnosis and does not replace laboratory testing "
    "or professional medical advice."
)


# =========================
# SCREENING FORM
# =========================

with st.form("screening_form"):

    st.markdown(
        '<p class="section-title">❤️ Health Conditions</p>',
        unsafe_allow_html=True
    )

    high_bp = st.selectbox(
        "Have you ever been told you have high blood pressure?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    high_chol = st.selectbox(
        "Have you ever been told you have high cholesterol?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    chol_check = st.selectbox(
        "Have you had your cholesterol checked in the past 5 years?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    stroke = st.selectbox(
        "Have you ever had a stroke?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    heart = st.selectbox(
        "Have you ever been told you had coronary heart disease or a heart attack?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    st.divider()

    st.markdown(
        '<p class="section-title">🏃 Lifestyle & Physical Activity</p>',
        unsafe_allow_html=True
    )

    bmi = st.number_input(
        "Body Mass Index (BMI)",
        min_value=10.0,
        max_value=90.0,
        value=27.0,
        step=0.5
    )

    smoker = st.selectbox(
        "Have you smoked at least 100 cigarettes in your entire life?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    phys_activity = st.selectbox(
        "Have you participated in physical activities in the past 30 days?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    fruits = st.selectbox(
        "Do you consume fruit one or more times per day?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    veggies = st.selectbox(
        "Do you consume vegetables one or more times per day?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    alcohol = st.selectbox(
        "Are you a heavy drinker (dataset definition)?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    st.divider()

    st.markdown(
        '<p class="section-title">🧠 General Health</p>',
        unsafe_allow_html=True
    )

    gen_health = st.select_slider(
        "In general, would you say your health is:",
        options=[1, 2, 3, 4, 5],
        value=3,
        format_func=lambda x: {
            1: "Excellent",
            2: "Very good",
            3: "Good",
            4: "Fair",
            5: "Poor"
        }[x]
    )

    ment_health = st.number_input(
        "Days during the past 30 days when mental health was not good",
        min_value=0,
        max_value=30,
        value=0
    )

    phys_health = st.number_input(
        "Days during the past 30 days when physical health was not good",
        min_value=0,
        max_value=30,
        value=0
    )

    diff_walk = st.selectbox(
        "Do you have serious difficulty walking or climbing stairs?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    healthcare = st.selectbox(
        "Do you have any kind of health care coverage?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    no_doc_cost = st.selectbox(
        "Could not see a doctor in the past 12 months because of cost?",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    st.divider()

    st.markdown(
        '<p class="section-title">👤 Personal Information</p>',
        unsafe_allow_html=True
    )

    sex = st.selectbox(
        "Sex category as encoded in the dataset",
        [0, 1],
        format_func=lambda x: "Female (0)" if x == 0 else "Male (1)"
    )

    age = st.select_slider(
        "Age category",
        options=list(range(1, 14)),
        value=8,
        format_func=lambda x: {
            1: "18–24", 2: "25–29", 3: "30–34",
            4: "35–39", 5: "40–44", 6: "45–49",
            7: "50–54", 8: "55–59", 9: "60–64",
            10: "65–69", 11: "70–74",
            12: "75–79", 13: "80 or older"
        }[x]
    )

    education = st.select_slider(
        "Education level",
        options=list(range(1, 7)),
        value=4,
        format_func=lambda x: {
            1: "Never attended / kindergarten",
            2: "Grades 1–8",
            3: "Grades 9–11",
            4: "High school graduate",
            5: "Some college / technical school",
            6: "College graduate"
        }[x]
    )

    income = st.select_slider(
        "Income category",
        options=list(range(1, 9)),
        value=5,
        format_func=lambda x: {
            1: "Less than $10,000",
            2: "$10,000–$14,999",
            3: "$15,000–$19,999",
            4: "$20,000–$24,999",
            5: "$25,000–$34,999",
            6: "$35,000–$49,999",
            7: "$50,000–$74,999",
            8: "$75,000 or more"
        }[x]
    )

    st.divider()

    submitted = st.form_submit_button(
        "🩺 Estimate Screening Result"
    )


# =========================
# PREDICTION
# =========================

if submitted:

    input_data = {
        "HighBP": high_bp,
        "HighChol": high_chol,
        "CholCheck": chol_check,
        "BMI": bmi,
        "Smoker": smoker,
        "Stroke": stroke,
        "HeartDiseaseorAttack": heart,
        "PhysActivity": phys_activity,
        "Fruits": fruits,
        "Veggies": veggies,
        "HvyAlcoholConsump": alcohol,
        "AnyHealthcare": healthcare,
        "NoDocbcCost": no_doc_cost,
        "GenHlth": gen_health,
        "MentHlth": ment_health,
        "PhysHlth": phys_health,
        "DiffWalk": diff_walk,
        "Sex": sex,
        "Age": age,
        "Education": education,
        "Income": income
    }

    input_df = pd.DataFrame([input_data])[features]

    probability = float(
        model.predict_proba(input_df)[0, 1]
    )

    predicted_class = int(probability >= threshold)

    st.divider()

    st.markdown(
        '<p class="section-title">📊 Your Screening Results</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">
                MODEL SCORE FOR DIABETES CLASS
            </div>
            <div class="result-score">
                {probability * 100:.1f}%
            </div>
            <div class="result-label">
                Decision Threshold: {threshold:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if predicted_class == 1:

        st.error("⚠️ Higher Model-Estimated Screening Risk")

        st.write(
            "The model score is at or above the selected threshold. "
            "Consider discussing appropriate screening or testing "
            "with a healthcare professional."
        )

    else:

        st.success("✓ Lower Model-Estimated Screening Risk")

        st.write(
            "The model score is below the selected threshold. "
            "This does not rule out diabetes. Consult a healthcare "
            "professional if you have concerns or symptoms."
        )

    st.info(
        "Important: This percentage is a model score, not a confirmed "
        "medical probability or diagnosis."
    )

    st.caption(f"Model used: {model_name}")


# =========================
# FOOTER
# =========================

st.markdown("""
<div class="footer">
    Diabetes Risk Screening Tool<br>
    Machine Learning Project | Educational Use Only
</div>
""", unsafe_allow_html=True)
