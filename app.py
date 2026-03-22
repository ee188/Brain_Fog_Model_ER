import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Page setup
# -----------------------------------
st.set_page_config(
    page_title="Brain Fog Risk Estimate",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------------
# Minimal styling
# -----------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.main-title {
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}

.subtext {
    font-size: 1.05rem;
    color: #666;
    max-width: 900px;
    margin-bottom: 1.4rem;
    line-height: 1.6;
}

.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    margin-top: 1.2rem;
    margin-bottom: 0.35rem;
}

.result-box {
    padding: 0.3rem 0 0.7rem 0;
    margin-top: 0.8rem;
}

.result-label {
    font-size: 1.5rem;
    font-weight: 650;
    margin-bottom: 0.25rem;
}

.result-prob {
    font-size: 1.05rem;
    color: #444;
    margin-bottom: 0.45rem;
}

.low {
    color: #2e7d32;
}

.moderate {
    color: #a66a00;
}

.high {
    color: #b00020;
}

.small-note {
    font-size: 0.92rem;
    color: #777;
    line-height: 1.5;
}

.soft-box {
    background: #f7f7f7;
    border-radius: 0.9rem;
    padding: 1rem 1rem;
    margin-top: 0.7rem;
    margin-bottom: 1rem;
}

.sidebar-note {
    font-size: 0.9rem;
    color: #777;
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
            "Low likelihood",
            "low",
            "This suggests a relatively low likelihood of reported memory or concentration difficulty."
        )
    elif prob < 0.50:
        return (
            "Moderate likelihood",
            "moderate",
            "This suggests a moderate likelihood of reported memory or concentration difficulty."
        )
    else:
        return (
            "High likelihood",
            "high",
            "This suggests a higher likelihood of reported memory or concentration difficulty."
        )

# -----------------------------------
# Header
# -----------------------------------
st.markdown('<div class="main-title">Brain Fog Risk Estimate</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtext">This educational tool estimates the likelihood of reported confusion or memory difficulty using health, sleep, mood, and medication-related inputs. It is designed as a healthcare informatics prototype based on public survey data.</div>',
    unsafe_allow_html=True
)

top_left, top_right = st.columns([2.2, 1])

with top_left:
    st.markdown(
        """
        <div class="soft-box">
        <strong>What this tool does</strong><br>
        It estimates a statistical risk based on patterns found in survey data. It does not diagnose brain fog, identify a cause, or recommend treatment.
        </div>
        """,
        unsafe_allow_html=True
    )

with top_right:
    with st.expander("How to use this tool"):
        st.write(
            """
            1. Enter your health and sleep information in the sidebar.  
            2. Choose the symptom and income categories that fit best.  
            3. Check any medication classes that apply.  
            4. Click **See my result** to view the estimate.  
            """
        )
        st.write(
            "The result is an educational risk estimate, not a medical opinion."
        )

# -----------------------------------
# Sidebar inputs
# -----------------------------------
st.sidebar.markdown("## Inputs")
st.sidebar.markdown('<div class="sidebar-note">Adjust the information below, then click <strong>See my result</strong>.</div>', unsafe_allow_html=True)

st.sidebar.markdown("### Personal and health")
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
    help="Biologic sex category used in the source survey data."
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
    help="Education level grouped using the original survey categories."
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
    help="This is a simplified version of the income-to-poverty ratio used by the model."
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

st.sidebar.markdown("### Sleep and mood")
sleep_hours = st.sidebar.slider(
    "Average sleep per night (hours)",
    min_value=2.0,
    max_value=14.0,
    value=7.0,
    step=0.5,
    help="Typical number of hours slept per night."
)

phq_category = st.sidebar.selectbox(
    "Depression symptom level",
    options=[
        "Minimal (0–4)",
        "Mild (5–9)",
        "Moderate (10–14)",
        "Moderately severe (15–19)",
        "Severe (20–27)"
    ],
    help="Based on PHQ-9 severity categories. Higher levels reflect more depressive symptoms."
)
phq9_total = map_phq_category(phq_category)

st.sidebar.markdown("### Medication burden")
med_count = st.sidebar.slider(
    "Number of current medications",
    min_value=0,
    max_value=22,
    value=1,
    help="Approximate number of medications currently taken."
)

st.sidebar.markdown("### Medication classes")
st.sidebar.caption("Check any classes that apply.")

is_benzo = st.sidebar.checkbox("Benzodiazepine", help="Examples: lorazepam, alprazolam, clonazepam")
is_antidepressant = st.sidebar.checkbox("Antidepressant", help="Examples: sertraline, fluoxetine, duloxetine")
is_antipsychotic = st.sidebar.checkbox("Antipsychotic", help="Examples: quetiapine, risperidone, olanzapine")
is_sedative = st.sidebar.checkbox("Sedative / sleep medication", help="Examples: zolpidem, eszopiclone")
is_anticholinergic = st.sidebar.checkbox("Anticholinergic medication", help="Examples: diphenhydramine, oxybutynin")
is_opioid = st.sidebar.checkbox("Opioid pain medication", help="Examples: hydrocodone, tramadol, oxycodone")
is_anticonvulsant = st.sidebar.checkbox("Anticonvulsant / nerve pain medication", help="Examples: gabapentin, pregabalin")
is_muscle_relaxant = st.sidebar.checkbox("Muscle relaxant", help="Examples: cyclobenzaprine, methocarbamol")
is_steroid = st.sidebar.checkbox("Steroid", help="Examples: prednisone, methylprednisolone")
is_stimulant = st.sidebar.checkbox("Stimulant", help="Examples: methylphenidate, amphetamine")

predict_clicked = st.sidebar.button("See my result", use_container_width=True)

# -----------------------------------
# Main explanations
# -----------------------------------
st.markdown('<div class="section-title">What the inputs mean</div>', unsafe_allow_html=True)

explain_col1, explain_col2 = st.columns(2)

with explain_col1:
    st.write("**Income level**")
    st.caption(
        "This app uses broad income categories instead of asking you to enter a technical ratio. Lower categories represent fewer financial resources."
    )

    st.write("**Depression symptom level**")
    st.caption(
        "This is based on PHQ-9 severity groupings. Higher categories reflect greater symptom burden."
    )

with explain_col2:
    st.write("**Medication burden**")
    st.caption(
        "Taking 5 or more medications is often called polypharmacy in health research."
    )

    st.write("**Medication classes**")
    st.caption(
        "These classes represent medication groups that may relate to mood, sedation, concentration, or cognitive symptoms."
    )

st.markdown("---")

# -----------------------------------
# Prediction
# -----------------------------------
st.markdown('<div class="section-title">Your result</div>', unsafe_allow_html=True)

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
    risk_label, color_class, message = risk_label_and_message(prob)

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label {color_class}">{risk_label}</div>
            <div class="result-prob">Estimated probability: {prob:.1%}</div>
            <div>{message}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "This estimate reflects a statistical pattern from survey data. It should not be used to diagnose memory problems, determine a cause, or guide treatment decisions."
    )

    with st.expander("Show the values used by the model"):
        st.dataframe(input_df, use_container_width=True)

else:
    st.markdown(
        '<div class="small-note">Use the sidebar to enter information, then click <strong>See my result</strong>.</div>',
        unsafe_allow_html=True
    )

# -----------------------------------
# Footer
# -----------------------------------
st.markdown("---")
st.caption(
    "Educational healthcare informatics prototype built with public NHANES survey data and machine learning."
)
