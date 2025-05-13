import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import cohere

# Load model and data
model = joblib.load("gb_model.pkl")
data = pd.read_csv("expenses.csv")
user_db_path = "users.csv"

st.set_page_config(page_title="Medical Expense Predictor", layout="wide")

# -------------------- Styling --------------------
def set_login_bg():
    st.markdown("""
        <style>
        .stApp {
            background-image: url("https://thumbs.dreamstime.com/b/paper-cut-health-insurance-icon-isolated-black-background-patient-protection-security-safety-protect-concept-art-style-vector-190439983.jpg");
            background-size: cover;
            background-attachment: fixed;
        }
        </style>
    """, unsafe_allow_html=True)

def set_main_bg():
    st.markdown("""
        <style>
        .stApp {
            background-image: url("https://wallpaperaccess.com/full/6890234.jpg");
            background-size: cover;
            background-attachment: fixed;
        }
        </style>
    """, unsafe_allow_html=True)

def set_global_text_style():
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        font-size: 20px !important;
        font-weight: bold !important;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stSubheader, .stTitle {
        color: white !important;
        font-size: 26px !important;
    }
    label, .stSlider > div > div {
        color: white !important;
        font-size: 18px !important;
    }
    .stTextInput > div > div > input,
    .stNumberInput > div > input,
    .stSelectbox > div > div > div,
    .stSlider > div {
        color: black !important;
        background-color: rgba(255, 255, 255, 0.7) !important;
        font-size: 18px !important;
        border-radius: 8px;
    }
    .stButton > button {
        font-size: 18px !important;
        font-weight: bold;
        color: white !important;
        background-color: #2b6cb0 !important;
        border-radius: 8px;
        border: none;
        padding: 8px 20px;
    }
    .stDataFrame, .stTable {
        color: white !important;
        background-color: rgba(0, 0, 0, 0.6) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------- Auth --------------------
def load_users():
    if not os.path.exists(user_db_path):
        pd.DataFrame(columns=["username", "password"]).to_csv(user_db_path, index=False)
    return pd.read_csv(user_db_path)

def login(username, password):
    users = load_users()
    return not users[(users["username"] == username) & (users["password"] == password)].empty

def signup(username, password):
    users = load_users()
    pd.DataFrame([[username, password]], columns=["username", "password"]).to_csv(user_db_path, mode='a', header=False, index=False)
    return True

# -------------------- Pages --------------------
def login_page():
    st.image("banner.png", use_column_width=True)
    st.title("\U0001F512 Login / Sign Up")
    tab1, tab2 = st.tabs(["\U0001F513 Login", "\U0001F4DD Sign Up"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials.")

    with tab2:
        st.markdown("""
**Password must meet all the following conditions:**
- 8 to 16 characters in length
- At least one uppercase letter (A-Z)
- At least one number (0-9)
- At least one special character (e.g., !@#$%^&*)
- Must match the 'Re-type Password' field
""")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Re-type Password", type="password")

        def is_valid_password(pw):
            import re
            if len(pw) < 8 or len(pw) > 16:
                return "Password must be 8–16 characters long."
            if not re.search(r"[A-Z]", pw):
                return "Password must include at least one uppercase letter."
            if not re.search(r"[0-9]", pw):
                return "Password must include at least one number."
            if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", pw):
                return "Password must include at least one special character."
            return True

        if st.button("Register"):
            if not new_username or not new_password or not confirm_password:
                st.warning("All fields are required.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                result = is_valid_password(new_password)
                if result is not True:
                    st.error(result)
                elif new_username in load_users()["username"].values:
                    st.warning("Username already exists.")
                else:
                    signup(new_username, new_password)
                    st.success("Registration successful! Please login.")

def logout_button():
    if st.button("\U0001F6AA Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

def show_raw_data():
    st.subheader("\U0001F4C4 Raw Data")
    st.dataframe(data)
    st.markdown("### \U0001F4CA Statistical Summary")
    st.dataframe(data.describe())

def show_visualizations():
    st.subheader("\U0001F4CA Data Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots()
        sns.boxplot(x="smoker", y="charges", data=data, ax=ax1)
        ax1.set_title("Charges by Smoker Status")
        ax1.set_xlabel("Smoker")
        ax1.set_ylabel("Charges")
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots()
        sns.barplot(x="region", y="charges", data=data, estimator=np.mean, ax=ax2)
        ax2.set_title("Average Charges by Region")
        ax2.set_xlabel("Region")
        ax2.set_ylabel("Average Charges")
        st.pyplot(fig2)

    st.markdown("### \U0001F4C8 Custom Chart")
    x_axis = st.selectbox("Choose X-axis", data.columns)
    y_axis = st.selectbox("Choose Y-axis", data.columns)
    plot_type = st.selectbox("Select Plot Type", ["scatter", "line", "bar", "box"])

    fig3, ax3 = plt.subplots()
    try:
        if plot_type == "scatter":
            sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax3)
        elif plot_type == "line":
            sns.lineplot(data=data, x=x_axis, y=y_axis, ax=ax3)
        elif plot_type == "bar":
            sns.barplot(data=data, x=x_axis, y=y_axis, ax=ax3)
        elif plot_type == "box":
            sns.boxplot(data=data, x=x_axis, y=y_axis, ax=ax3)
        ax3.set_xlabel(x_axis)
        ax3.set_ylabel(y_axis)
        ax3.set_title(f"{plot_type.title()} Plot: {x_axis} vs {y_axis}")
        st.pyplot(fig3)
    except Exception as e:
        st.error(f"Could not generate plot: {e}")

    st.markdown("### \U0001F525 Correlation Matrix Heatmap")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f", ax=ax4)
    ax4.set_title("Correlation Matrix Heatmap")
    st.pyplot(fig4)

def show_prediction():
    st.subheader("\U0001F9E0 Predict Medical Charges")
    age = st.slider("Age", 18, 100, 25)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", 10.0, 50.0, 22.5)
    children = st.slider("Children", 0, 5, 0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

    # Save to session
    st.session_state.update({"age": age, "sex": sex, "bmi": bmi, "children": children, "smoker": smoker, "region": region})

    sex = 1 if sex == "male" else 0
    smoker = 1 if smoker == "yes" else 0
    region_map = {"southeast": 0, "southwest": 1, "northeast": 2, "northwest": 3}
    region = region_map[region]

    if st.button("Predict"):
        input_data = np.array([[age, sex, bmi, children, smoker, region]])
        prediction = model.predict(input_data)[0]
        st.success(f"\U0001F4B0 Predicted Medical Charges: ${prediction:,.2f}")

def cohere_chat():
    st.subheader("\U0001F916 Ask Cohere AI Based on Your Profile")
    profile_info = f"Here is my profile: Age: {st.session_state.get('age', 'N/A')}, Gender: {st.session_state.get('sex', 'N/A')}, BMI: {st.session_state.get('bmi', 'N/A')}, Children: {st.session_state.get('children', 'N/A')}, Smoker: {st.session_state.get('smoker', 'N/A')}, Region: {st.session_state.get('region', 'N/A')}."

    questions = [
        "What kind of diet should I follow based on my health profile?",
        "Am I at risk of lifestyle diseases?",
        "What preventive health check-ups should I consider?",
        "How can I reduce my future medical expenses?"
    ]

    for q in questions:
        if st.button(q):
            try:
                co = cohere.Client(st.secrets["COHERE_API_KEY"])
                response = co.chat(message=profile_info + " " + q)
                st.success(getattr(response, 'text', 'No response'))
            except Exception as e:
                st.error(f"Cohere API call failed: {e}")

    user_q = st.text_area("Or type your own question:")
    if st.button("Ask Cohere") and user_q.strip():
        try:
            co = cohere.Client(st.secrets["COHERE_API_KEY"])
            response = co.chat(message=profile_info + " " + user_q)
            st.success(getattr(response, 'text', 'No response'))
        except Exception as e:
            st.error(f"Cohere API call failed: {e}")

# -------------------- Page Logic --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

set_global_text_style()

if not st.session_state.logged_in:
    set_login_bg()
    login_page()
else:
    set_main_bg()
    logout_button()
    tabs = st.tabs(["\U0001F4C4 Raw Data", "\U0001F4CA Visualizations", "\U0001F9E0 Prediction", "\U0001F916 Chat"])
    with tabs[0]: show_raw_data()
    with tabs[1]: show_visualizations()
    with tabs[2]: show_prediction()
    with tabs[3]: cohere_chat()
