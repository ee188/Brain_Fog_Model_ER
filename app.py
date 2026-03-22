import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Page setup
# -----------------------------------
st.set_page_config(
    page_title="Brain Fog Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------
# Custom styling
# -----------------------------------
st.markdown("""
<style>
.main-title {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}
.subtext {
    font-size: 1.1rem;
    color: #555;
    margin-bottom: 1.25rem;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.result-card {
    padding: 1.25rem;
    border-radius: 1rem;
    margin-top: 1rem;
    margin-bottom: 1rem;
    border: 1px solid #e6e6e6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.low-risk {
    background-color: #eef9f0;
    border-left: 8px solid #2e8b57;
}
.moderate-risk {
    background-color: #fff8e6;
    border-left: 8px solid #d4a017;
}
.high-risk {
    background-color: #fff0f0;
    border-left: 8px solid #c0392b;
}
.small-note {
    font-size: 0.95rem;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Load model and columns
# -----------------------------------
model = joblib.load("brain_fog_model_v3.pkl")
model_columns = joblib.load("brain_fog_model_columns_v3.pkl")

# -----------------------------------
# Helpers
# -----------------------------------
def map_income_category(cat: str) -> float:
    mapping = {
        "Below poverty line": 0.8,
        "Around poverty line": 1.0,
        "Lower income": 1.5,
        "Middle income": 2.5,
        "Higher income": 4.0
    }
    return mapping[cat]

def map_phq_category(cat: str) -> int:
    mapping = {
        "Minimal (0–4)": 2,
        "Mild (5–9)": 7,
        "Moderate (10–14)": 12,
        "Moderately severe (15–19)": 17,
        "Severe (20–27)": 23
    }
    return mapping[cat]

def build_input_df(
    age,
    income_ratio,
    bmi,
    sleep_hours,
    phq9_total,
    med_count,
    sex,
    education,
    is_benzo,
    is_antidepressant,
    is_antipsychotic,
    is_sedative,
    is_anticholinergic,
    is_opioid,
    is_anticonvulsant,
    is_muscle_relaxant,
    is_steroid,
    is_stimulant
):
    polypharmacy = 1 if med_count >= 5 else 0
    short_sleep = 1 if sleep_hours < 7 else 0
    sex_male = 1 if sex == "Male" else 0

    input_data = {
        "age": age,
        "income_ratio": income_ratio,
        "bmi": bmi,
        "sleep_hours": sleep_hours,
        "short_sleep": short_sleep,
        "phq9_total": phq9_total,
        "med_count": med_count,
        "polypharmacy": polypharmacy,
        "is_benzo": int(is_benzo),
        "is_antidepressant": int(is_antidepressant),
        "is_antipsychotic": int(is_antipsychotic),
        "is_sedative": int(is_sedative),
        "is_anticholinergic": int(is_anticholinergic),
        "is_opioid": int(is_opioid),
        "is_anticonvulsant": int(is_anticonvulsant),
        "is_muscle_relaxant": int(is_muscle_relaxant),
        "is_steroid": int(is_steroid),
        "is_stimulant": int(is_stimulant),
        "sex_Male": sex_male,
        "education_2.0": 1 if education == "9th–11th grade" else 0,
        "education_3.0": 1 if education == "High school / GED" else 0,
        "education_4.0": 1 if education == "Some college / Associate degree" else 0,
        "education_5.0": 1 if education == "College graduate or above" else 0,
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    return input_df

def risk_label_and_message(prob: float):
    if prob < 0.20:
        return (
            "Low",
            "The model estimates a relatively low likelihood of reported confusion or memory difficulty.",
            "low-risk"
        )
    elif prob < 0.50:
        return (
            "Moderate",
            "The model estimates a moderate likelihood of reported confusion or memory difficulty.",
            "moderate-risk"
        )
    else:
        return (
            "High",
            "The model estimates a higher likelihood of reported confusion or memory difficulty.",
            "high-risk"
        )

# -----------------------------------
# Header / main content
# -----------------------------------
st.markdown('<div class="main-title">🧠 Brain Fog Risk Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtext">An educational healthcare informatics demo that estimates the probability of reported confusion or memory difficulty using health, sleep, mood, and medication-related inputs.</div>',
    unsafe_allow_html=True
)

col_a, col_b = st.columns([2, 1])

with col_a:
    st.info(
        "This tool is an educational prototype based on public survey data. "
        "It is not a diagnosis, not medical advice, and not a prescribing tool."
    )

with col_b:
    with st.expander("How to use this tool"):
        st.write(
            """
            1. Enter basic health and sleep information in the sidebar.
            2. Choose a depression symptom category based on PHQ-9 severity.
            3. Estimate the number of medications you take.
            4. Check any medication classes that apply.
            5. Click **Estimate Brain Fog Risk** to view the model output.
            """
        )
        st.write(
            "The result is a probability estimate, not a clinical decision."
        )

# -----------------------------------
# Sidebar inputs
# -----------------------------------
st.sidebar.header("Enter Inputs")

st.sidebar.subheader("Personal and Health Information")

age = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=80,
    value=40,
    help="Age in years."
)

sex = st.sidebar.radio(
    "Sex",
    options=["Female", "Male"],
    help="Biologic sex category used in the original survey data."
)

education = st.sidebar.selectbox(
    "Highest education level",
    options=[
        "Less than 9th grade",
        "9th–11th grade",
        "High school / GED",
        "Some college / Associate degree",
        "College graduate or above"
    ],
    help="Education level as grouped in the NHANES survey."
)

income_category = st.sidebar.selectbox(
    "Household income level",
    options=[
        "Below poverty line",
        "Around poverty line",
        "Lower income",
        "Middle income",
        "Higher income"
    ],
    help="This uses income-to-poverty ratio behind the scenes."
)

income_ratio = map_income_category(income_category)

bmi = st.sidebar.slider(
    "BMI",
    min_value=14.0,
    max_value=60.0,
    value=28.0,
    step=0.1,
    help="Body Mass Index, a rough measure based on height and weight."
)

st.sidebar.subheader("Sleep and Mood")

sleep_hours = st.sidebar.slider(
    "Average sleep per night (hours)",
    min_value=2.0,
    max_value=14.0,
    value=7.0,
    step=0.5,
    help="Typical number of hours slept per night."
)

phq_category = st.sidebar.selectbox(
    "Depression symptom level (PHQ-9 category)",
    options=[
        "Minimal (0–4)",
        "Mild (5–9)",
        "Moderate (10–14)",
        "Moderately severe (15–19)",
        "Severe (20–27)"
    ],
    help="Higher categories indicate more depressive symptoms."
)

phq9_total = map_phq_category(phq_category)

st.sidebar.subheader("Medication Burden")

med_count = st.sidebar.slider(
    "Number of current medications",
    min_value=0,
    max_value=22,
    value=1,
    help="Approximate number of medications currently taken."
)

st.sidebar.subheader("Medication Classes")
st.sidebar.caption("Check all that apply.")

is_benzo = st.sidebar.checkbox("Benzodiazepine", help="Examples: lorazepam, alprazolam, clonazepam")
is_antidepressant = st.sidebar.checkbox("Antidepressant", help="Examples: sertraline, fluoxetine, duloxetine")
is_antipsychotic = st.sidebar.checkbox("Antipsychotic", help="Examples: quetiapine, olanzapine, risperidone")
is_sedative = st.sidebar.checkbox("Sedative / sleep medication", help="Examples: zolpidem, eszopiclone")
is_anticholinergic = st.sidebar.checkbox("Anticholinergic medication", help="Examples: diphenhydramine, oxybutynin")
is_opioid = st.sidebar.checkbox("Opioid pain medication", help="Examples: hydrocodone, tramadol, oxycodone")
is_anticonvulsant = st.sidebar.checkbox("Anticonvulsant / nerve pain medication", help="Examples: gabapentin, pregabalin")
is_muscle_relaxant = st.sidebar.checkbox("Muscle relaxant", help="Examples: cyclobenzaprine, methocarbamol")
is_steroid = st.sidebar.checkbox("Steroid", help="Examples: prednisone, methylprednisolone")
is_stimulant = st.sidebar.checkbox("Stimulant", help="Examples: methylphenidate, amphetamine")

predict_clicked = st.sidebar.button("Estimate Brain Fog Risk", use_container_width=True)

# -----------------------------------
# Main page explanatory content
# -----------------------------------
st.markdown('<div class="section-title">What the inputs mean</div>', unsafe_allow_html=True)

exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    st.markdown("**Income level**")
    st.caption(
        "This app uses a simplified version of income-to-poverty ratio. "
        "Lower values represent fewer financial resources."
    )

    st.markdown("**PHQ-9 category**")
    st.caption(
        "PHQ-9 is a common depression symptom questionnaire. "
        "Higher categories reflect greater symptom burden."
    )

with exp_col2:
    st.markdown("**Medication burden**")
    st.caption(
        "Taking 5 or more medications is often considered polypharmacy in healthcare research."
    )

    st.markdown("**Medication classes**")
    st.caption(
        "These checkboxes represent medication groups that may relate to cognition, sedation, or mood."
    )

# -----------------------------------
# Prediction section
# -----------------------------------
st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

if predict_clicked:
    input_df = build_input_df(
        age=age,
        income_ratio=income_ratio,
        bmi=bmi,
        sleep_hours=sleep_hours,
        phq9_total=phq9_total,
        med_count=med_count,
        sex=sex,
        education=education,
        is_benzo=is_benzo,
        is_antidepressant=is_antidepressant,
        is_antipsychotic=is_antipsychotic,
        is_sedative=is_sedative,
        is_anticholinergic=is_anticholinergic,
        is_opioid=is_opioid,
        is_anticonvulsant=is_anticonvulsant,
        is_muscle_relaxant=is_muscle_relaxant,
        is_steroid=is_steroid,
        is_stimulant=is_stimulant,
    )

    prob = model.predict_proba(input_df)[0][1]
    risk_label, interpretation, css_class = risk_label_and_message(prob)

    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <h3 style="margin-top:0;">Estimated Brain Fog Risk: {risk_label}</h3>
            <p style="font-size:1.15rem; margin-bottom:0.5rem;"><strong>Predicted probability:</strong> {prob:.1%}</p>
            <p style="margin-bottom:0;">{interpretation}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning(
        "This result reflects a statistical pattern from survey data. "
        "It should not be used to diagnose cognitive problems or guide treatment."
    )

    with st.expander("Show input values used by the model"):
        st.dataframe(input_df, use_container_width=True)

else:
    st.caption("Use the sidebar to enter values, then click **Estimate Brain Fog Risk**.")

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.caption(
    "Built as an educational healthcare informatics prototype using public NHANES survey data and machine learning."
)
