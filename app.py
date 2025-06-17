import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
import os

#st.set_page_config(    page_title="Multi-Disease Predictor",    layout="wide")


# ---------- UTILITY FUNCTION ----------
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def set_background(image_file):
    bg_image = get_base64_image(image_file)
 
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bg_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .main {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 12px;
            border: 1px solid #ddd;
        }}
        
        </style>
        """, unsafe_allow_html=True)


# ---------- LOAD MODELS ----------
@st.cache_resource
def load_resources():
    return {
        "diabetes_model": pickle.load(open("models/diabetes_model.sav", "rb")),
        "heart_model": pickle.load(open("models/heart_disease_model.sav", "rb")),
        "lung_model": pickle.load(open("models/lung_cancer_model.sav", "rb")),
        "diabetes_scaler": pickle.load(open("scaler/diabetes_scaler.pkl", "rb")),
        "heart_scaler": pickle.load(open("scaler/heart_disease_scaler.pkl", "rb")),
        "lung_scaler": pickle.load(open("scaler/lung_cancer_scaler.pkl", "rb"))
    }

resources = load_resources()

# ---------- CUSTOM FONT + BUTTON STYLE ----------
st.markdown("""
    <style>
    /* Import bold Poppins font */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    /* Global font and text styling */
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        font-size: 20px;
        font-weight: 500;
        color: #1e1e1e !important;
        line-height: 1.7;
    }

    /* Fix the translucent white box */
    .main {
        background-color: transparent !important;
        padding: 0 !important;
        border-radius: 0 !important;
        border: none !important;
    }

    /* Kill white overlay boxes if created by Streamlit */
    div[style*="background-color: rgba(255, 255, 255"] {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* Style the buttons */
    .stButton > button {
        background-color: #ff9800;
        color: white;
        font-weight: bold;
        font-size: 16px;
        border: none;
        padding: 0.6em 1.2em;
        border-radius: 6px;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background-color: #ffa726;
        transform: scale(1.02);
        cursor: pointer;
    }
    
        main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        margin: auto;
    }
    .main {
        padding: 0rem 1rem;
        max-width: 100vw;
    }
   
    </style>
""", unsafe_allow_html=True)



# ---------- HOME PAGE ----------
def home():
    set_background("images/home_bgg.jpeg")
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.title("📈Multi-Disease Predictor Model Using ML & streamlit ")

    # About
    st.subheader("👨‍⚕️ About us ")
    st.write("""
    This app is designed to help you better understand your potential health risks using modern AI techniques. 
    Whether you're monitoring your health proactively or checking for warning signs, our **AI-powered models** give you fast and clear results.
    
    Built using real medical datasets, the system analyzes your inputs and uses trained algorithms to provide risk predictions—all from the comfort of your home.
    """)

    # Diseases Covered
    st.subheader("🩺 Diseases Covered ")
    st.markdown("""
    ### 1️⃣ Diabetes Prediction  
    - Based on inputs like glucose level, BMI, age, and more  
    - Uses a Support Vector Machine (SVM) model for precise risk detection  

    ### 2️⃣ Heart Disease Prediction  
    - Considers chest pain, cholesterol, blood pressure, and related indicators  
    - Uses Logistic Regression for evaluating heart disease probability  

    ### 3️⃣ Lung Cancer Prediction  
    - Uses factors like smoking habits, coughing, and fatigue  
    - Trained with real symptom data to spot early signs using Logistic Regression
    """)
    # How It Works (Detailed)
    st.subheader("🔬 How It Works ")
    st.write("""
    Here's how the prediction process happens behind the scenes:

    **Step 1: Input Your Details**  
    You start by entering your basic health and lifestyle information such as age, smoking habits, blood pressure, or blood sugar levels—depending on the disease you're checking.

    **Step 2: Behind-the-Scenes AI**  
    Once submitted, your data is sent to a machine learning model trained specifically for that disease. The models were built using thousands of real patient records to ensure reliable predictions.

    **Step 3: Instant Prediction**  
    The model processes your data and makes a prediction in real time. It tells you whether you're likely at risk or not.

    **Step 4: Easy-to-Understand Output**  
    You’ll see a simple result like "At Risk" or "Not at Risk"—no medical jargon, just actionable insights. 
    """)

    # Summary
    st.subheader("📋 Summary of Benefits")
    st.markdown("""
    ✅ Understand potential health risks early  
    ✅ Take preventive action based on AI insights  
    ✅ Clean, simple interface for everyone—no medical background needed  
    ✅ Free to use and privacy-focused  
    """)

    # Disclaimer
    #st.subheader("🧠 Note")
    #st.write("""
    #This tool is intended for **educational and awareness purposes only**.  
    #It is not a replacement for professional medical diagnosis.  
    #Always consult a qualified doctor for medical advice or treatment.
    #""")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- PREDICTION PAGE ----------
def predict():
    selected = st.sidebar.selectbox("🔬 Choose Disease to Predict", ["Diabetes", "Heart Disease", "Lung Cancer"], key="predict_select")

    if selected == "Diabetes":
        set_background("images/diabetes_bgg.jpeg")
        st.markdown('<div class="main">', unsafe_allow_html=True)
        st.header("🩸 Diabetes Prediction")
        col1, col2 = st.columns([2, 2])
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20)
            glucose = st.number_input("Glucose", 0, 300)
            bloodpressure = st.number_input("Blood Pressure", 0, 200)
            skinthickness = st.number_input("Skin Thickness", 0, 100)
        with col2:
            insulin = st.number_input("Insulin", 0, 900)
            bmi = st.number_input("BMI", 0.0, 70.0, step=0.1)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, step=0.01)
            age = st.number_input("Age", 1, 120)

        if st.button("🔍 Predict Diabetes"):
            try:
                data = pd.DataFrame([[pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age]],
                    columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
                scaled_input = resources["diabetes_scaler"].transform(data)
                result = resources["diabetes_model"].predict(scaled_input)
                st.success("✅ Diabetic" if result[0] == 1 else "❎ Not Diabetic")
            except Exception as e:
                st.error(f"Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    elif selected == "Heart Disease":
        set_background("images/heart_bg.jpeg")
        st.markdown('<div class="main">', unsafe_allow_html=True)
        st.header("❤️ Heart Disease Prediction")
        col1, col2 = st.columns([2, 2])
        with col1:
            age = st.number_input("Age", 1, 120)
            sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
            cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
            trestbps = st.number_input("Resting BP", 80, 200)
            chol = st.number_input("Cholesterol", 100, 600)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
            restecg = st.selectbox("Rest ECG", [0, 1, 2])
        with col2:
            thalach = st.number_input("Max Heart Rate", 60, 220)
            exang = st.selectbox("Exercise Induced Angina", [0, 1])
            oldpeak = st.number_input("Oldpeak", 0.0, 10.0, step=0.1)
            slope = st.selectbox("Slope", [0, 1, 2])
            ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
            thal = st.selectbox("Thal", [0, 1, 2, 3])

        if st.button("💓 Predict Heart Disease"):
            try:
                data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                                      thalach, exang, oldpeak, slope, ca, thal]],
                    columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                             "thalach", "exang", "oldpeak", "slope", "ca", "thal"])
                scaled_input = resources["heart_scaler"].transform(data)
                result = resources["heart_model"].predict(scaled_input)
                st.success("🚨 Heart Disease Detected" if result[0] == 1 else "✅ No Heart Disease")
            except Exception as e:
                st.error(f"Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    elif selected == "Lung Cancer":
        set_background("images/lung_bgg.jpeg")
        st.markdown('<div class="main">', unsafe_allow_html=True)
        st.header("🫁 Lung Cancer Prediction")
        col1, col2 = st.columns([2, 2])
        # inside your Lung Cancer block...

        # mapping for all yes/no features
        yes_no = {"No": 1, "Yes": 2}

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            gender = 0 if gender == "Female" else 1

            age = st.number_input("Age", 1, 120, value=50)

            smoking = st.selectbox("Do you smoke?", list(yes_no.keys()))
            smoking = yes_no[smoking]

            yellow_fingers = st.selectbox("Yellow fingers?", list(yes_no.keys()))
            yellow_fingers = yes_no[yellow_fingers]

            anxiety = st.selectbox("Feel anxious?", list(yes_no.keys()))
            anxiety = yes_no[anxiety]

            peer_pressure = st.selectbox("Peer pressure?", list(yes_no.keys()))
            peer_pressure = yes_no[peer_pressure]

            chronic_disease = st.selectbox("Chronic disease?", list(yes_no.keys()))
            chronic_disease = yes_no[chronic_disease]

            fatigue = st.selectbox("Fatigue?", list(yes_no.keys()))
            fatigue = yes_no[fatigue]

        with col2:
            allergy = st.selectbox("Allergies?", list(yes_no.keys()))
            allergy = yes_no[allergy]

            wheezing = st.selectbox("Wheezing?", list(yes_no.keys()))
            wheezing = yes_no[wheezing]

            alcohol_consuming = st.selectbox("Alcohol consumption?", list(yes_no.keys()))
            alcohol_consuming = yes_no[alcohol_consuming]

            coughing = st.selectbox("Coughing?", list(yes_no.keys()))
            coughing = yes_no[coughing]

            shortness_of_breath = st.selectbox("Shortness of breath?", list(yes_no.keys()))
            shortness_of_breath = yes_no[shortness_of_breath]

            swallowing_difficulty = st.selectbox("Swallowing difficulty?", list(yes_no.keys()))
            swallowing_difficulty = yes_no[swallowing_difficulty]

            chest_pain = st.selectbox("Chest pain?", list(yes_no.keys()))
            chest_pain = yes_no[chest_pain]



        if st.button("🔍 Predict Lung Cancer"):
            try:
                data = pd.DataFrame([[gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                                      chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
                                      coughing, shortness_of_breath, swallowing_difficulty, chest_pain]],
                    columns=['gender', 'age', 'smoking', 'YELLOW_FINGERS', 'anxiety', 'PEER_PRESSURE',
                             'chronic_disease', 'FATIGUE', 'ALLERGY', 'wheezing', 'alcohol_consuming',
                             'coughing', 'shortness_of_breath', 'swallowing_difficulty', 'chest_pain'])
                scaled_input = resources["lung_scaler"].transform(data)
                result = resources["lung_model"].predict(scaled_input)
                if result[0] == 1: 
                    st.markdown(
                        '<p style="color: #D32F2F; font-size: 20px; font-weight: bold;">🚨 Lung Cancer Risk Detected</p>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                          '<p style="color: #D32F2F; font-size: 20px; font-weight: bold;">✅ No Lung Cancer Risk</p>',
                           unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Error: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

#---------Feedback-------------
def feedback():
    set_background("images/feedback_bg.jpeg")  # optional custom background
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.title("💬 Feedback !")

    # You can use st.form to batch inputs + a submit button
    with st.form("feedback_form"):
        name = st.text_input("Your Name (optional)")
        email = st.text_input("Your Email (optional)")
        rating = st.slider("How would you rate this app?", 1, 5, 3)
        comments = st.text_area("What can we improve?")
        submitted = st.form_submit_button("Submit Feedback")

    if submitted:
        # 1) Append to a CSV
        feedback_entry = {
            "timestamp": pd.Timestamp.now(),
            "name": name,
            "email": email,
            "rating": rating,
            "comments": comments
        }
        df = pd.DataFrame([feedback_entry])
        df.to_csv("feedback.csv", mode="a", header=not os.path.exists("feedback.csv"), index=False)

        # 2) Give user a thank-you message
        st.success("🙏 Thanks for your feedback!")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- SIDEBAR NAVIGATION ----------
if 'page' not in st.session_state:
    st.session_state.page = "Home"

page = st.sidebar.selectbox("📍 Navigate", ["Home", "Predict","Feedback"], index=["Home", "Predict","Feedback"].index(st.session_state.page))
st.session_state.page = page

if page == "Home":
    home()
elif page == "Predict":
    predict()
elif page == "Feedback":
    feedback()


# ---------- FOOTER ----------
st.markdown("---")
st.markdown("© 2025 Multi-Disease Predictor | tneha1025@ gmail")
