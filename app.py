from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "models" / "gallstone_model.joblib"
DATA_PATH = ROOT_DIR / "data" / "gallstone.csv"
TARGET_COLUMN = "Gallstone Status"
EXPECTED_FEATURE_COUNT = 38

FEATURE_GROUPS = (
    (
        "اطلاعات فردی",
        ("Age", "Gender"),
    ),
    (
        "سوابق بیماری و شرایط همراه",
        (
            "Comorbidity",
            "Coronary Artery Disease (CAD)",
            "Hypothyroidism",
            "Hyperlipidemia",
            "Diabetes Mellitus (DM)",
        ),
    ),
    (
        "ترکیب بدن و شاخص‌های چربی",
        (
            "Height",
            "Weight",
            "Body Mass Index (BMI)",
            "Total Body Water (TBW)",
            "Extracellular Water (ECW)",
            "Intracellular Water (ICW)",
            "Extracellular Fluid/Total Body Water (ECF/TBW)",
            "Total Body Fat Ratio (TBFR) (%)",
            "Lean Mass (LM) (%)",
            "Body Protein Content (Protein) (%)",
            "Visceral Fat Rating (VFR)",
            "Bone Mass (BM)",
            "Muscle Mass (MM)",
            "Obesity (%)",
            "Total Fat Content (TFC)",
            "Visceral Fat Area (VFA)",
            "Visceral Muscle Area (VMA) (Kg)",
            "Hepatic Fat Accumulation (HFA)",
        ),
    ),
    (
        "آزمایش‌های خون و شاخص‌های بالینی",
        (
            "Glucose",
            "Total Cholesterol (TC)",
            "Low Density Lipoprotein (LDL)",
            "High Density Lipoprotein (HDL)",
            "Triglyceride",
            "Aspartat Aminotransferaz (AST)",
            "Alanin Aminotransferaz (ALT)",
            "Alkaline Phosphatase (ALP)",
            "Creatinine",
            "Glomerular Filtration Rate (GFR)",
            "C-Reactive Protein (CRP)",
            "Hemoglobin (HGB)",
            "Vitamin D",
        ),
    ),
)

FEATURE_LABELS = {
    "Age": "سن",
    "Gender": "جنس (کدگذاری‌شده)",
    "Comorbidity": "بیماری‌های همراه (کدگذاری‌شده)",
    "Coronary Artery Disease (CAD)": "بیماری عروق کرونر",
    "Hypothyroidism": "کم‌کاری تیروئید",
    "Hyperlipidemia": "چربی خون بالا",
    "Diabetes Mellitus (DM)": "دیابت",
    "Height": "قد",
    "Weight": "وزن",
    "Body Mass Index (BMI)": "شاخص توده بدنی",
    "Total Body Water (TBW)": "آب کل بدن",
    "Extracellular Water (ECW)": "آب خارج‌سلولی",
    "Intracellular Water (ICW)": "آب داخل‌سلولی",
    "Extracellular Fluid/Total Body Water (ECF/TBW)": "نسبت آب خارج‌سلولی به آب کل بدن",
    "Total Body Fat Ratio (TBFR) (%)": "نسبت چربی کل بدن",
    "Lean Mass (LM) (%)": "توده بدون چربی",
    "Body Protein Content (Protein) (%)": "میزان پروتئین بدن",
    "Visceral Fat Rating (VFR)": "شاخص چربی احشایی",
    "Bone Mass (BM)": "توده استخوانی",
    "Muscle Mass (MM)": "توده عضلانی",
    "Obesity (%)": "شاخص چاقی",
    "Total Fat Content (TFC)": "میزان چربی کل",
    "Visceral Fat Area (VFA)": "سطح چربی احشایی",
    "Visceral Muscle Area (VMA) (Kg)": "سطح عضله احشایی",
    "Hepatic Fat Accumulation (HFA)": "تجمع چربی کبدی",
    "Glucose": "گلوکز",
    "Total Cholesterol (TC)": "کلسترول کل",
    "Low Density Lipoprotein (LDL)": "کلسترول LDL",
    "High Density Lipoprotein (HDL)": "کلسترول HDL",
    "Triglyceride": "تری‌گلیسرید",
    "Aspartat Aminotransferaz (AST)": "آنزیم AST",
    "Alanin Aminotransferaz (ALT)": "آنزیم ALT",
    "Alkaline Phosphatase (ALP)": "آلکالین فسفاتاز (ALP)",
    "Creatinine": "کراتینین",
    "Glomerular Filtration Rate (GFR)": "نرخ فیلتراسیون گلومرولی",
    "C-Reactive Protein (CRP)": "پروتئین واکنشی C (CRP)",
    "Hemoglobin (HGB)": "هموگلوبین",
    "Vitamin D": "ویتامین D",
}

BINARY_YES_NO_FEATURES = {
    "Coronary Artery Disease (CAD)",
    "Hypothyroidism",
    "Hyperlipidemia",
    "Diabetes Mellitus (DM)",
}


st.set_page_config(
    page_title="پیش‌بینی سنگ کیسه صفرا",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        direction: rtl;
        font-family: Tahoma, Arial, sans-serif;
    }
    [data-testid="stMarkdownContainer"] {
        direction: rtl;
        text-align: right;
    }
    [data-testid="stWidgetLabel"] {
        direction: rtl;
        text-align: right;
    }
    .stNumberInput input,
    [data-baseweb="select"] > div {
        direction: rtl;
        text-align: right;
    }
    [data-testid="stMetric"] {
        direction: rtl;
        text-align: right;
        background: rgba(49, 51, 63, 0.04);
        border-radius: 0.75rem;
        padding: 1rem;
    }
    .block-container {
        max-width: 1150px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"فایل مدل پیدا نشد: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("مدل نام ویژگی‌های ورودی را ارائه نمی‌کند.")
    if not hasattr(model, "predict_proba"):
        raise ValueError("مدل بارگذاری‌شده از predict_proba پشتیبانی نمی‌کند.")

    feature_names = list(model.feature_names_in_)
    if len(feature_names) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"مدل باید {EXPECTED_FEATURE_COUNT} ویژگی داشته باشد؛ "
            f"{len(feature_names)} ویژگی پیدا شد."
        )
    if set(model.classes_) != {0, 1}:
        raise ValueError(f"کلاس‌های مدل باید ۰ و ۱ باشند؛ {model.classes_} پیدا شد.")

    return model, feature_names


@st.cache_data
def load_dataset():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"فایل دیتاست پیدا نشد: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


model, feature_names = load_model()
dataset = load_dataset()

if TARGET_COLUMN not in dataset.columns:
    raise ValueError(f"ستون هدف «{TARGET_COLUMN}» در دیتاست پیدا نشد.")

dataset_features = [column for column in dataset.columns if column != TARGET_COLUMN]
if dataset_features != feature_names:
    raise ValueError(
        "نام یا ترتیب ویژگی‌های دیتاست با مدل مطابقت ندارد. "
        "ستون‌های دیتاست و مدل را بررسی کنید."
    )
if len(FEATURE_LABELS) != EXPECTED_FEATURE_COUNT:
    raise ValueError("برای همهٔ ویژگی‌های مدل برچسب فارسی تعریف نشده است.")

grouped_features = [feature for _, features in FEATURE_GROUPS for feature in features]
if grouped_features != feature_names:
    raise ValueError("گروه‌بندی ورودی‌ها با ترتیب ویژگی‌های مدل مطابقت ندارد.")


def render_feature_input(feature):
    series = dataset[feature]
    label = FEATURE_LABELS[feature]
    help_text = f"نام ویژگی در مدل: {feature}"
    unique_values = set(series.dropna().unique().tolist())

    if unique_values == {0, 1}:
        default = int(series.mode(dropna=True).iloc[0])
        if feature in BINARY_YES_NO_FEATURES:
            format_option = lambda value: "بله" if value == 1 else "خیر"
        else:
            format_option = lambda value: str(value)
        return int(
            st.selectbox(
                label,
                options=[0, 1],
                index=[0, 1].index(default),
                format_func=format_option,
                help=help_text,
            )
        )

    if pd.api.types.is_integer_dtype(series.dtype):
        default = int(series.median())
        return int(
            st.number_input(
                label,
                value=default,
                step=1,
                help=help_text,
            )
        )

    default = float(series.median())
    return float(
        st.number_input(
            label,
            value=default,
            step=0.1,
            format="%.2f",
            help=help_text,
        )
    )


st.title("سامانه پیش‌بینی احتمال وجود سنگ کیسه صفرا")
st.markdown(
    "مقادیر فرم را وارد کنید. مقدار اولیهٔ فیلدهای عددی از میانهٔ دیتاست و "
    "مقدار اولیهٔ فیلدهای دودویی از پرتکرارترین مقدار آن‌ها تعیین شده است."
)
st.warning(
    "این سامانه صرفاً آموزشی و پژوهشی است و ابزار تشخیص پزشکی نیست. "
    "نتیجهٔ آن جایگزین نظر پزشک یا آزمایش‌های تشخیصی نمی‌شود."
)

with st.form("gallstone_prediction_form"):
    input_values = {}
    for section_title, section_features in FEATURE_GROUPS:
        st.subheader(section_title)
        columns = st.columns(2)
        for index, feature in enumerate(section_features):
            with columns[index % 2]:
                input_values[feature] = render_feature_input(feature)

    submitted = st.form_submit_button(
        "محاسبهٔ پیش‌بینی",
        type="primary",
        use_container_width=True,
    )

if submitted:
    input_frame = pd.DataFrame(
        [[input_values[feature] for feature in feature_names]],
        columns=feature_names,
    )
    prediction = int(model.predict(input_frame)[0])
    class_probabilities = model.predict_proba(input_frame)[0]
    probabilities = {
        int(class_value): float(probability)
        for class_value, probability in zip(model.classes_, class_probabilities)
    }

    if set(probabilities) != {0, 1}:
        raise ValueError("احتمال هر دو کلاس ۰ و ۱ از مدل دریافت نشد.")

    st.subheader("نتیجهٔ پیش‌بینی")
    if prediction == 0:
        st.error("پیش‌بینی مدل: کلاس ۰ — دارای سنگ کیسه صفرا")
    else:
        st.success("پیش‌بینی مدل: کلاس ۱ — بدون سنگ کیسه صفرا")

    class_zero, class_one = st.columns(2)
    with class_zero:
        st.metric("احتمال کلاس ۰ — دارای سنگ", f"{probabilities[0]:.1%}")
        st.progress(probabilities[0])
    with class_one:
        st.metric("احتمال کلاس ۱ — بدون سنگ", f"{probabilities[1]:.1%}")
        st.progress(probabilities[1])
