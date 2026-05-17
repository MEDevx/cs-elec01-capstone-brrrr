import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
import time

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="USM — Student Retention EWS",
    layout="wide",
    page_icon="assets/usm_seal.png",
    initial_sidebar_state="expanded",
)

# ── LOAD LOGO ─────────────────────────────────────────────────────────────────
def get_base64_image(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

logo_b64 = get_base64_image("assets/usm_seal.png")

# ── CRITICAL CSS INJECTION ────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="collapsedControl"] {
    background-color: #0e1f12 !important;
    border: 1px solid #f9a825 !important;
    border-radius: 8px !important;
    z-index: 100 !important;
}
[data-testid="collapsedControl"]:hover {
    background-color: #1b5e20 !important;
}
[data-testid="collapsedControl"] svg {
    fill: #f9a825 !important;
    color: #f9a825 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
""", unsafe_allow_html=True)

try:
    with open("style.css", encoding="utf-8") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ── RANDOM FOREST ──────────────────────────────
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.split_counts = {}

    def fit(self, X, y):
        self.split_counts = {i: 0 for i in range(X.shape[1])}
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        if depth >= self.max_depth or n_labels <= 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common_label(y))
        feat_idxs = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        if best_feat is None:
            return Node(value=self._most_common_label(y))
        self.split_counts[best_feat] += 1
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (len(left_idxs) / n) * e_l + (len(right_idxs) / n) * e_r
        return parent_entropy - child_entropy

    def _entropy(self, y):
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        if len(y) == 0: return 0
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node(): return node.value
        if x[node.feature] <= node.threshold: return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class CustomRandomForest:
    def __init__(self, n_trees=5, max_depth=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.classes_ = []
        self.feature_importances_ = None

    def fit(self, X, y):
        self.trees = []
        self.classes_ = np.unique(y)
        self.feature_importances_ = np.zeros(X.shape[1])
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth)
            X_samp, y_samp = self._bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)
            for feat, count in tree.split_counts.items():
                self.feature_importances_[feat] += count
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ = self.feature_importances_ / np.sum(self.feature_importances_)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        return np.array([np.bincount(pred).argmax() for pred in tree_preds])

    def predict_proba(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        probas = []
        for sample_preds in tree_preds:
            counts = {c: 0 for c in self.classes_}
            for p in sample_preds: counts[p] += 1
            total = sum(counts.values())
            probas.append([counts[c]/total for c in self.classes_])
        return np.array(probas)


# ── DYNAMIC DATA PREPARATION & TRAINING ──────────────────────────────────────
@st.cache_resource
def prepare_and_train():
    try:
        df = pd.read_csv("data.csv", sep=";")
        df.columns = df.columns.str.strip()
        if "Target" not in df.columns:
            st.error("⚠️ Column 'Target' not found.")
            st.stop()
        feature_cols = [col for col in df.columns if col != "Target"]
        target_mapping = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
        df['Target'] = df['Target'].str.strip().map(target_mapping)
        df = df.dropna()
        y = df['Target'].astype(int).values
        X = df[feature_cols].values
        if len(y) > 2000:
            np.random.seed(42)
            sample_idx = np.random.choice(len(y), 2000, replace=False)
            X, y = X[sample_idx], y[sample_idx]
    except Exception as e:
        st.error(f"⚠️ Error loading data.csv: {e}")
        st.stop()

    model = CustomRandomForest(n_trees=5, max_depth=6)
    model.fit(X, y)
    y_pred = model.predict(X)
    accuracy = np.sum(y == y_pred) / len(y)
    
    # Compute class distribution
    class_dist = {0: int(np.sum(y==0)), 1: int(np.sum(y==1)), 2: int(np.sum(y==2))}
    return model, accuracy, feature_cols, df, class_dist

rf_model, model_accuracy, FEATURE_COLS, df_raw, class_dist = prepare_and_train()
LABEL_DECODER = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

# ── CATEGORICAL MAPPINGS ─────────────────────────────────────────────────────
MARITAL_STATUS = {
    "Single": 1, "Married": 2, "Widower": 3,
    "Divorced": 4, "Facto Union / Live-in": 5, "Legally Separated": 6
}
APPLICATION_MODE = {
    "Regular Admission (Freshman)": 1, "Transferee": 42,
    "Cross-Enrollee": 43, "Change of Course": 51, "Irregular Student": 17,
}
COURSE = {
    "BS Agriculture": 9003, "BS Agronomy": 9070, "BS Horticulture": 9085,
    "BS Agricultural and Biosystems Engineering": 9119, "BS Civil Engineering": 9130,
    "BS Computer Engineering": 9147, "BS Electronics Engineering": 9238,
    "BS Computer Science": 9254, "BS Information Systems": 9500,
    "BS Nursing": 9556, "BS Midwifery": 9670, "BS Pharmacy": 9773,
    "BS Nutrition and Dietetics": 9853, "Bachelor of Elementary Education (BEEd)": 8014,
    "Bachelor of Early Childhood Education": 171, "Bachelor of Secondary Education (BSEd)": 33,
    "BS Forestry": 9991, "BS Environmental Science": 9238,
    "Bachelor of Library and Information Science": 9119, "Doctor of Veterinary Medicine (DVM)": 9130,
}
PREV_QUALIFICATION = {
    "Senior High School Graduate (Academic Track)": 1,
    "Senior High School Graduate (TVL Track)": 9,
    "Senior High School Graduate (Arts & Design / Sports)": 12,
    "ALS Passer": 10, "TESDA / TVET Certificate": 39,
    "College Undergraduate (did not finish)": 6, "Bachelor's Degree Graduate": 2,
    "Bachelor's Degree (Different Field)": 3, "Master's Degree": 4, "Doctorate": 5,
    "Elementary Graduate (old curriculum)": 11, "High School Graduate (old curriculum)": 14,
}
QUALIFICATION_LEVEL = {
    "Elementary Graduate": 35, "High School Graduate (old curriculum)": 14,
    "Senior High School Graduate": 1, "ALS Passer": 10,
    "TESDA / TVET Certificate (NC I–III)": 39,
    "College Undergraduate (did not finish)": 6, "Bachelor's Degree": 2,
    "Bachelor's Degree (different field)": 3, "Post-Baccalaureate / Professional Course": 41,
    "Master's Degree": 4, "Doctorate": 5, "Can't Read or Write": 35, "Unknown / Not Stated": 34,
}
OCCUPATION = {
    "Student / Not Working": 0, "Farmer / Fisherfolk": 6,
    "Skilled Agricultural / Forestry Worker": 163, "Government Employee (Rank & File)": 4,
    "Government Employee (Supervisor/Manager)": 1, "Private Employee (Rank & File)": 8,
    "Private Employee (Supervisor/Manager)": 112, "Teacher / Education Professional": 123,
    "Health Professional (Doctor/Nurse/Midwife)": 122, "Engineer / Technician": 121,
    "OFW / Overseas Worker": 183, "Self-Employed / Small Business Owner": 152,
    "Driver / Transport Worker": 182, "Construction Worker / Laborer": 171,
    "Factory / Assembly Worker": 181, "Domestic Helper / Household Service Worker": 151,
    "Market Vendor / Street Seller": 195, "Armed Forces / PNP": 101,
    "Retired / Not Working": 90, "Deceased / Unknown": 99,
}
BINARY = {"No": 0, "Yes": 1}
GENDER = {"Female": 0, "Male": 1}
APPLICATION_ORDER = {"1st Choice Program": 1, "2nd Choice Program": 2}
HOUSEHOLD_INCOME = {
    "Class E — Below ₱10,000 / month": 0.5,
    "Class D — ₱10,000 to ₱20,000 / month": 1.2,
    "Class C — ₱20,000 to ₱50,000 / month": 2.5,
    "Class B — ₱50,000 to ₱100,000 / month": 4.0,
    "Class A — Above ₱100,000 / month": 6.0,
}

# ── TOP NAV ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="usm-nav">
  <div class="usm-logo-ring">
    {f'<img src="data:image/png;base64,{logo_b64}" alt="USM Seal">' if logo_b64 else '<span style="font-size:20px;">🌿</span>'}
  </div>
  <div>
    <div class="usm-nav-title">University of Southern Mindanao</div>
    <div class="usm-nav-sub">Student Retention Early Warning System</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── DYNAMIC SIDEBAR ───────────────────────────────────────────────────────────
def group_label(text):
    st.markdown(f'<p style="font-size: 11px; font-weight: 700; color: #69f07a; letter-spacing: 1.5px; text-transform: uppercase; margin: 18px 0 8px; padding-bottom: 4px; border-bottom: 1px solid rgba(255, 255, 255, 0.05);">{text}</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown(f"""
    <div class="sidebar-header">
      <div class="sidebar-icon-wrap">
        <i class="ti ti-user-circle" style="font-size: 22px; color: var(--gold);"></i>
      </div>
      <div>
        <div class="sidebar-title">Student Profile</div>
        <div class="sidebar-sub">Input Parameters</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    user_inputs = {}

    def cat_input(label, mapping, col_name):
        default_val = int(df_raw[col_name].mode()[0]) if col_name in df_raw.columns else list(mapping.values())[0]
        default_key = next((k for k, v in mapping.items() if v == default_val), list(mapping.keys())[0])
        choice = st.selectbox(label, list(mapping.keys()), index=list(mapping.keys()).index(default_key))
        return mapping[choice]

    def num_input(label, col_name, step=1.0, fmt="%.0f"):
        default = float(df_raw[col_name].mean()) if col_name in df_raw.columns else 0.0
        return st.number_input(label, value=round(default, 2), step=step, format=fmt)

    group_label("Student Identity")
    student_name = st.text_input("Student Name", placeholder="e.g. Juan Dela Cruz (optional)")
    student_id   = st.text_input("Student ID", placeholder="e.g. 2024-0192 (optional)")

    group_label("Demographics & Enrollment")
    user_inputs["Marital status"] = cat_input("Marital Status", MARITAL_STATUS, "Marital status")
    user_inputs["Gender"] = cat_input("Gender", GENDER, "Gender")
    user_inputs["Age at enrollment"] = int(num_input("Age at Enrollment", "Age at enrollment"))
    user_inputs["Nacionality"] = 1
    user_inputs["International"] = 0
    user_inputs["Displaced"] = cat_input("Living Away from Home / Relocated for Studies", BINARY, "Displaced")

    group_label("Application Details")
    user_inputs["Application mode"] = cat_input("Application Mode", APPLICATION_MODE, "Application mode")
    user_inputs["Application order"] = cat_input("Application Order", APPLICATION_ORDER, "Application order")
    user_inputs["Course"] = cat_input("Course", COURSE, "Course")
    attendance_col = next((c for c in FEATURE_COLS if "attendance" in c.lower()), "Daytime/evening attendance\t")
    user_inputs[attendance_col] = 1

    group_label("Academic Background")
    user_inputs["Previous qualification"] = cat_input("Previous Qualification", PREV_QUALIFICATION, "Previous qualification")
    raw_prev_grade = st.number_input(
        "Senior High School GWA (75 = Passing, 100 = Perfect)",
        min_value=75.0, max_value=100.0, value=85.0, step=0.5, format="%.1f"
    )
    user_inputs["Previous qualification (grade)"] = round((raw_prev_grade - 75) / 25 * 200, 2)
    raw_admission = st.number_input(
        "USMCEE Score (%) — minimum varies by program: 45%–65%",
        min_value=0.0, max_value=100.0, value=55.0, step=0.5, format="%.1f"
    )
    user_inputs["Admission grade"] = round(raw_admission * 2, 2)

    group_label("Family Background")
    user_inputs["Mother's qualification"] = cat_input("Mother's Qualification", QUALIFICATION_LEVEL, "Mother's qualification")
    user_inputs["Father's qualification"] = cat_input("Father's Qualification", QUALIFICATION_LEVEL, "Father's qualification")
    user_inputs["Mother's occupation"] = cat_input("Mother's Occupation", OCCUPATION, "Mother's occupation")
    user_inputs["Father's occupation"] = cat_input("Father's Occupation", OCCUPATION, "Father's occupation")

    group_label("Student Status & Support")
    user_inputs["Educational special needs"] = cat_input("Special Educational Needs", BINARY, "Educational special needs")
    user_inputs["Debtor"] = cat_input("Debtor", BINARY, "Debtor")
    user_inputs["Tuition fees up to date"] = cat_input("Has Unpaid School Fees", BINARY, "Tuition fees up to date")
    user_inputs["Scholarship holder"] = cat_input("Scholarship Holder", BINARY, "Scholarship holder")

    group_label("1st Semester Performance")
    user_inputs["Curricular units 1st sem (credited)"] = int(num_input("Units Credited", "Curricular units 1st sem (credited)"))
    user_inputs["Curricular units 1st sem (enrolled)"] = int(num_input("Units Enrolled", "Curricular units 1st sem (enrolled)"))
    user_inputs["Curricular units 1st sem (evaluations)"] = int(num_input("Units Evaluated", "Curricular units 1st sem (evaluations)"))
    user_inputs["Curricular units 1st sem (approved)"] = int(num_input("Units Approved", "Curricular units 1st sem (approved)"))
    raw_grade_1 = st.number_input(
        "1st Sem Grade Average (1.0 = Highest, 3.0 = Lowest Pass, 5.0 = Failed)",
        min_value=1.0, max_value=5.0, value=2.0, step=0.25, format="%.2f", key="grade_1st"
    )
    user_inputs["Curricular units 1st sem (grade)"] = round((5.0 - raw_grade_1) / 4.0 * 20, 2)
    user_inputs["Curricular units 1st sem (without evaluations)"] = int(num_input("Units Without Evaluation", "Curricular units 1st sem (without evaluations)"))

    group_label("2nd Semester Performance")
    user_inputs["Curricular units 2nd sem (credited)"] = int(num_input("Units Credited", "Curricular units 2nd sem (credited)"))
    user_inputs["Curricular units 2nd sem (enrolled)"] = int(num_input("Units Enrolled", "Curricular units 2nd sem (enrolled)"))
    user_inputs["Curricular units 2nd sem (evaluations)"] = int(num_input("Units Evaluated", "Curricular units 2nd sem (evaluations)"))
    user_inputs["Curricular units 2nd sem (approved)"] = int(num_input("Units Approved", "Curricular units 2nd sem (approved)"))
    raw_grade_2 = st.number_input(
        "2nd Sem Grade Average (1.0 = Highest, 3.0 = Lowest Pass, 5.0 = Failed)",
        min_value=1.0, max_value=5.0, value=2.0, step=0.25, format="%.2f", key="grade_2nd"
    )
    user_inputs["Curricular units 2nd sem (grade)"] = round((5.0 - raw_grade_2) / 4.0 * 20, 2)
    user_inputs["Curricular units 2nd sem (without evaluations)"] = int(num_input("Units Without Evaluation", "Curricular units 2nd sem (without evaluations)"))

    group_label("Economic Context")
    user_inputs["Unemployment rate"] = num_input("Regional Unemployment Rate (%)", "Unemployment rate", step=0.1, fmt="%.1f")
    user_inputs["Inflation rate"] = num_input("Regional Inflation Rate (%)", "Inflation rate", step=0.1, fmt="%.1f")
    income_label = st.selectbox("Estimated Household Monthly Income", list(HOUSEHOLD_INCOME.keys()))
    user_inputs["GDP"] = HOUSEHOLD_INCOME[income_label]

    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("Run Analysis", use_container_width=True, type="primary")

# ── LIVE PREDICTION LOGIC ─────────────────────────────────────────────────────
try:
    input_data = np.array([[user_inputs[col] for col in FEATURE_COLS]])
except KeyError as e:
    st.error(f"⚠️ Feature mismatch error. Expected column not found in inputs: {e}")
    st.stop()

raw_pred = rf_model.predict(input_data)[0]
prediction = LABEL_DECODER.get(raw_pred, "Unknown")
probabilities = rf_model.predict_proba(input_data)[0]
confidence = max(probabilities) * 100 if len(probabilities) > 0 else 0

dropout_prob  = probabilities[0] * 100 if len(probabilities) > 0 else 0
enrolled_prob = probabilities[1] * 100 if len(probabilities) > 1 else 0
graduate_prob = probabilities[2] * 100 if len(probabilities) > 2 else 0

# ── DERIVE ACADEMIC CONTEXT FROM INPUTS ──────────────────────────────────────
enrolled_1 = user_inputs.get("Curricular units 1st sem (enrolled)", 1)
approved_1 = user_inputs.get("Curricular units 1st sem (approved)", 0)
enrolled_2 = user_inputs.get("Curricular units 2nd sem (enrolled)", 1)
approved_2 = user_inputs.get("Curricular units 2nd sem (approved)", 0)
pass_rate_1 = round((approved_1 / enrolled_1) * 100) if enrolled_1 > 0 else 0
pass_rate_2 = round((approved_2 / enrolled_2) * 100) if enrolled_2 > 0 else 0
avg_pass_rate = round((pass_rate_1 + pass_rate_2) / 2)

grade_1_raw = raw_grade_1
grade_2_raw = raw_grade_2
sem_trend = grade_2_raw - grade_1_raw  # positive = worsening, negative = improving

is_debtor = user_inputs.get("Debtor", 0) == 1
has_scholarship = user_inputs.get("Scholarship holder", 0) == 1
is_displaced = user_inputs.get("Displaced", 0) == 1
has_special_needs = user_inputs.get("Educational special needs", 0) == 1

# Risk score (simple composite for display)
risk_flags = []
if dropout_prob > 40: risk_flags.append(("Low Approval Rate", "academic"))
if is_debtor: risk_flags.append(("Financial Debt", "financial"))
if sem_trend > 0.5: risk_flags.append(("Declining Grades", "academic"))
if pass_rate_1 < 60 or pass_rate_2 < 60: risk_flags.append(("Failed Units", "academic"))
if is_displaced: risk_flags.append(("Relocated for Study", "social"))
if has_special_needs: risk_flags.append(("Special Ed. Needs", "support"))
if not has_scholarship and dropout_prob > 30: risk_flags.append(("No Financial Aid", "financial"))

# ── HELPERS ───────────────────────────────────────────────────────────────────
COLOR_MAP = {"Dropout": "#ef5350", "Enrolled": "#ff9800", "Graduate": "#66bb6a", "Unknown": "#999999"}
CLASS_MAP = {"Dropout": "dropout", "Enrolled": "enrolled", "Graduate": "graduate", "Unknown": "unknown"}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans", color="#a5d6a7"),
    margin=dict(l=10, r=10, t=10, b=10),
)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PREDICTION VERDICT (Hero Banner)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
  <div class="section-num">1</div>
  <span class="section-title">Classification Result</span>
  <div class="section-rule"></div>
</div>
""", unsafe_allow_html=True)

color      = COLOR_MAP[prediction]
acc_display = round(model_accuracy * 100, 1)

# Risk level label
if prediction == "Dropout":
    risk_level, risk_cls, risk_icon = "HIGH RISK", "danger", "ti-alert-triangle"
elif prediction == "Enrolled":
    risk_level, risk_cls, risk_icon = "MODERATE RISK", "warning", "ti-alert-circle"
else:
    risk_level, risk_cls, risk_icon = "LOW RISK", "success", "ti-circle-check"

# Confidence ring color
conf_ring_pct = round(confidence)

# Build identity strings cleanly before injecting into HTML
name_html = f'<strong style="color:#fff; font-size:15px;">{student_name}</strong>' if student_name else '<span style="color:rgba(255,255,255,0.35); font-style:italic;">No name provided</span>'
id_html   = f'&nbsp;<span style="color:rgba(255,255,255,0.25);">·</span>&nbsp; <span style="color:rgba(255,255,255,0.5);">ID: {student_id}</span>' if student_id else ''

st.markdown(f"""
<div class="verdict-hero" style="border-color:{color}20; border-left: 4px solid {color};">
  <div class="verdict-left">
    <div class="verdict-tag {risk_cls}">
      <i class="ti {risk_icon}"></i> {risk_level}
    </div>
    <div class="verdict-outcome" style="color:{color};">{prediction}</div>
    <div class="verdict-sub">
      {name_html}{id_html}
      &nbsp;<span style="color:rgba(255,255,255,0.25);">·</span>&nbsp; {len(FEATURE_COLS)} features analysed
    </div>
    <div class="verdict-meta-row">
      <div class="verdict-meta-pill"><i class="ti ti-binary-tree-2"></i> Random Forest · 5 Trees · Depth 6</div>
      <div class="verdict-meta-pill"><i class="ti ti-crosshair"></i> Training Accuracy: {acc_display}%</div>
      <div class="verdict-meta-pill"><i class="ti ti-database"></i> n = {len(df_raw)} students</div>
    </div>
  </div>
  <div class="verdict-right">
    <div class="conf-donut-wrap">
      <svg width="110" height="110" viewBox="0 0 110 110">
        <circle cx="55" cy="55" r="46" fill="none" stroke="#162b19" stroke-width="10"/>
        <circle cx="55" cy="55" r="46" fill="none" stroke="{color}" stroke-width="10"
          stroke-dasharray="{round(conf_ring_pct * 2.89)} 289"
          stroke-dashoffset="72" stroke-linecap="round"/>
      </svg>
      <div class="conf-donut-label">
        <div class="conf-pct" style="color:{color};">{conf_ring_pct}%</div>
        <div class="conf-caption">confidence</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Probability Bars (3 classes inline) ──────────────────────────────────────
st.markdown("""
<div class="prob-row-wrap">
  <div class="prob-row-title">Class Probability Breakdown</div>
  <div class="prob-row">
""", unsafe_allow_html=True)

for label, prob, clr in [("Dropout", dropout_prob, "#ef5350"), ("Enrolled", enrolled_prob, "#ff9800"), ("Graduate", graduate_prob, "#66bb6a")]:
    active = "active" if label == prediction else ""
    st.markdown(f"""
    <div class="prob-col {active}" style="--clr:{clr}">
      <div class="prob-label">{label}</div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width:{prob:.1f}%; background:{clr};"></div>
      </div>
      <div class="prob-pct" style="color:{clr};">{prob:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — STUDENT RISK SNAPSHOT (KPI Strip + Flag Cards)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header" style="margin-top:18px;">
  <div class="section-num">2</div>
  <span class="section-title">Student Risk Snapshot</span>
  <div class="section-rule"></div>
</div>
""", unsafe_allow_html=True)

# KPI Strip
kpi1_color = "#ef5350" if pass_rate_1 < 60 else ("#ff9800" if pass_rate_1 < 80 else "#66bb6a")
kpi2_color = "#ef5350" if pass_rate_2 < 60 else ("#ff9800" if pass_rate_2 < 80 else "#66bb6a")
trend_icon = "ti-trending-down" if sem_trend > 0 else "ti-trending-up"
trend_color = "#ef5350" if sem_trend > 0 else "#66bb6a"
trend_label = f"+{sem_trend:.2f} (declining)" if sem_trend > 0 else f"{sem_trend:.2f} (improving)"

fin_status = "Debtor" if is_debtor else ("Scholar" if has_scholarship else "Self-Funded")
fin_color = "#ef5350" if is_debtor else ("#f9a825" if has_scholarship else "#78909c")

st.markdown(f"""
<div class="kpi-strip">
  <div class="kpi-tile">
    <div class="kpi-icon-wrap" style="background:#1a2e1b;"><i class="ti ti-school" style="color:#69f07a;"></i></div>
    <div class="kpi-body">
      <div class="kpi-val" style="color:{kpi1_color};">{pass_rate_1}%</div>
      <div class="kpi-lbl">1st Sem Pass Rate</div>
      <div class="kpi-sub">{approved_1} of {enrolled_1} units</div>
    </div>
  </div>
  <div class="kpi-tile">
    <div class="kpi-icon-wrap" style="background:#1a2e1b;"><i class="ti ti-school" style="color:#69f07a;"></i></div>
    <div class="kpi-body">
      <div class="kpi-val" style="color:{kpi2_color};">{pass_rate_2}%</div>
      <div class="kpi-lbl">2nd Sem Pass Rate</div>
      <div class="kpi-sub">{approved_2} of {enrolled_2} units</div>
    </div>
  </div>
  <div class="kpi-tile">
    <div class="kpi-icon-wrap" style="background:#1d1a0e;"><i class="ti {trend_icon}" style="color:{trend_color};"></i></div>
    <div class="kpi-body">
      <div class="kpi-val" style="color:{trend_color};">{grade_2_raw:.2f}</div>
      <div class="kpi-lbl">Grade Trend (2nd Sem)</div>
      <div class="kpi-sub">{trend_label}</div>
    </div>
  </div>
  <div class="kpi-tile">
    <div class="kpi-icon-wrap" style="background:#1a1a2e;"><i class="ti ti-wallet" style="color:{fin_color};"></i></div>
    <div class="kpi-body">
      <div class="kpi-val" style="color:{fin_color};">{fin_status}</div>
      <div class="kpi-lbl">Financial Status</div>
      <div class="kpi-sub">{'Unpaid balance on record' if is_debtor else ('Grant / scholarship active' if has_scholarship else 'No financial aid')}</div>
    </div>
  </div>
  <div class="kpi-tile">
    <div class="kpi-icon-wrap" style="background:#0d1f1f;"><i class="ti ti-flag-3" style="color:#ef5350;"></i></div>
    <div class="kpi-body">
      <div class="kpi-val" style="color:#ef5350;">{len(risk_flags)}</div>
      <div class="kpi-lbl">Risk Flags</div>
      <div class="kpi-sub">{'Critical — act now' if len(risk_flags) >= 3 else ('Monitor closely' if len(risk_flags) > 0 else 'None detected')}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Risk Flag Pills
if risk_flags:
    flag_tag_map = {"academic": ("tag-ac", "Academic"), "financial": ("tag-fin", "Financial"), "social": ("tag-eng", "Social"), "support": ("tag-sch", "Support")}
    pills_html = '<div class="flag-pills-wrap"><div class="flag-pills-label"><i class="ti ti-flag-3"></i> Active Risk Flags</div><div class="flag-pills">'
    for flag_name, flag_type in risk_flags:
        tag_cls, _ = flag_tag_map.get(flag_type, ("tag-ac", ""))
        pills_html += f'<span class="tag {tag_cls}">{flag_name}</span>'
    pills_html += "</div></div>"
    st.markdown(pills_html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL ANALYTICS (Feature Importances + Sem Performance Chart)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header" style="margin-top:18px;">
  <div class="section-num">3</div>
  <span class="section-title">Model Analytics</span>
  <div class="section-rule"></div>
</div>
""", unsafe_allow_html=True)

c3, c4 = st.columns([1.6, 1.4], gap="large")

with c3:
    st.markdown('<div class="usm-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><i class="ti ti-chart-bar"></i> Top Feature Importances</div>', unsafe_allow_html=True)

    imps = rf_model.feature_importances_
    top_n = 7
    top_indices = np.argsort(imps)[-top_n:]
    top_imps = imps[top_indices]
    raw_feats = np.array(FEATURE_COLS)[top_indices]
    clean_feats = [str(f).replace('Curricular units', 'CU').replace(' attendance\t', '').strip() for f in raw_feats]

    # Color bars by type
    bar_colors = []
    for f in raw_feats:
        fs = str(f).lower()
        if "grade" in fs or "approved" in fs or "enrolled" in fs or "curricular" in fs:
            bar_colors.append("#66bb6a")
        elif "gdp" in fs or "unemploy" in fs or "inflation" in fs or "tuition" in fs or "debtor" in fs:
            bar_colors.append("#f9a825")
        else:
            bar_colors.append("#64b5f6")

    fi_fig = go.Figure(go.Bar(
        x=top_imps * 100,
        y=clean_feats,
        orientation="h",
        marker_color=bar_colors,
        marker_line_width=0,
        marker_cornerradius=4,
        text=[f"{round(v * 100, 1)}%" for v in top_imps],
        textposition="outside",
        textfont=dict(size=11, color="#c8e6c9"),
    ))
    fi_fig.update_layout(
        xaxis=dict(
            ticksuffix="%",
            tickfont=dict(size=10, color="#a5d6a7"),
            gridcolor="rgba(255,255,255,0.04)",
            showgrid=True,
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#e8f5e9"),
            showgrid=False,
            automargin=True,
        ),
        height=280,
        margin=dict(l=10, r=50, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.28,
    )
    st.plotly_chart(fi_fig, use_container_width=True, config={"displayModeBar": False})

    # Legend
    st.markdown("""
    <div style="display:flex; gap:14px; margin-top:4px; padding: 0 4px;">
      <span style="font-size:10px; color:#a5d6a7;"><span style="color:#66bb6a;">●</span> Academic</span>
      <span style="font-size:10px; color:#a5d6a7;"><span style="color:#f9a825;">●</span> Economic</span>
      <span style="font-size:10px; color:#a5d6a7;"><span style="color:#64b5f6;">●</span> Demographic</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown('<div class="usm-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><i class="ti ti-chart-line"></i> Semester Performance Comparison</div>', unsafe_allow_html=True)

    sem_labels = ["1st Semester", "2nd Semester"]
    pass_vals  = [pass_rate_1, pass_rate_2]
    grade_vals = [grade_1_raw, grade_2_raw]

    sem_fig = go.Figure()
    sem_fig.add_trace(go.Bar(
        name="Pass Rate (%)",
        x=sem_labels,
        y=pass_vals,
        marker_color=["#43a047" if v >= 75 else "#e53935" for v in pass_vals],
        marker_cornerradius=5,
        yaxis="y1",
        offsetgroup=1,
        text=[f"{v}%" for v in pass_vals],
        textposition="outside",
        textfont=dict(size=12, color="#e8f5e9"),
    ))
    sem_fig.add_trace(go.Scatter(
        name="Grade Avg",
        x=sem_labels,
        y=grade_vals,
        mode="lines+markers+text",
        line=dict(color="#f9a825", width=2.5, dash="dot"),
        marker=dict(size=9, color="#f9a825", line=dict(color="#fff", width=1.5)),
        yaxis="y2",
        text=[f"{v:.2f}" for v in grade_vals],
        textposition="top center",
        textfont=dict(size=11, color="#f9a825"),
    ))
    sem_fig.update_layout(
        yaxis=dict(
            title="Pass Rate (%)", range=[0, 115],
            tickfont=dict(size=10, color="#a5d6a7"),
            gridcolor="rgba(255,255,255,0.04)",
            showgrid=True,
            ticksuffix="%",
        ),
        yaxis2=dict(
            title="Grade (1.0–5.0)", overlaying="y", side="right",
            range=[0.5, 6],
            tickfont=dict(size=10, color="#f9a825"),
            showgrid=False,
        ),
        xaxis=dict(tickfont=dict(size=12, color="#e8f5e9"), showgrid=False),
        legend=dict(
            font=dict(size=10, color="#c8e6c9"),
            bgcolor="rgba(0,0,0,0)",
            orientation="h", y=-0.18
        ),
        height=280,
        margin=dict(l=10, r=40, t=10, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        barmode="group",
        bargap=0.35,
    )
    st.plotly_chart(sem_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ── Student Summary + Radar ───────────────────────────────────────────────────
c5, c6 = st.columns([1, 2], gap="large")

with c5:
    age_val     = user_inputs.get("Age at enrollment", "—")
    gender_val  = "Male" if user_inputs.get("Gender", 0) == 1 else "Female"
    scholar_val = "Yes" if has_scholarship else "No"
    debtor_val  = "Yes" if is_debtor else "No"
    course_val  = next((k for k, v in COURSE.items() if v == user_inputs.get("Course")), "—")
    course_short = (course_val[:26] + "…") if len(course_val) > 26 else course_val
    displaced_val = "Yes" if is_displaced else "No"
    report_date = pd.Timestamp.now().strftime("%b %d, %Y")

    st.markdown(f"""
    <div class="usm-card" style="height:100%;">
      <div class="card-title"><i class="ti ti-id-badge-2"></i> Student Summary</div>
      <div class="summary-row">
        <span class="sum-lbl">Name</span>
        <span class="sum-val">{student_name if student_name else '<em style="color:rgba(255,255,255,0.25);">—</em>'}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Student ID</span>
        <span class="sum-val">{student_id if student_id else '<em style="color:rgba(255,255,255,0.25);">—</em>'}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Age</span>
        <span class="sum-val">{age_val}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Gender</span>
        <span class="sum-val">{gender_val}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Program</span>
        <span class="sum-val" title="{course_val}">{course_short}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Scholarship</span>
        <span class="sum-val" style="color:{'#69f07a' if scholar_val == 'Yes' else '#78909c'};">{scholar_val}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Debtor</span>
        <span class="sum-val" style="color:{'#ef5350' if debtor_val == 'Yes' else '#78909c'};">{debtor_val}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Displaced</span>
        <span class="sum-val" style="color:{'#ff9800' if displaced_val == 'Yes' else '#78909c'};">{displaced_val}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Report Date</span>
        <span class="sum-val">{report_date}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with c6:
    st.markdown('<div class="usm-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title"><i class="ti ti-radar"></i> Risk Factor Radar</div>', unsafe_allow_html=True)

    radar_categories = ["Academic\nPerformance", "Financial\nStability", "Social\nFactors", "Enrollment\nStrength", "Progression\nRate"]
    
    # Compute normalized scores (0–100, higher = better = lower risk)
    acad_score  = min(100, avg_pass_rate)
    fin_score   = 80 if has_scholarship else (20 if is_debtor else 55)
    social_score= 60 if is_displaced else (50 if has_special_needs else 80)
    enroll_score= min(100, round((enrolled_1 + enrolled_2) / 2 * 5))
    prog_score  = min(100, round(((approved_1 + approved_2) / max(enrolled_1 + enrolled_2, 1)) * 100))

    radar_vals = [acad_score, fin_score, social_score, enroll_score, prog_score]

    # Benchmark (average student)
    benchmark = [72, 60, 72, 65, 70]

    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=benchmark + [benchmark[0]],
        theta=radar_categories + [radar_categories[0]],
        mode="lines",
        name="Avg. Student",
        line=dict(color="rgba(255,255,255,0.2)", width=1.5, dash="dot"),
        fill="toself",
        fillcolor="rgba(255,255,255,0.04)",
    ))
    radar_fig.add_trace(go.Scatterpolar(
        r=radar_vals + [radar_vals[0]],
        theta=radar_categories + [radar_categories[0]],
        mode="lines+markers",
        name="This Student",
        line=dict(color=color, width=2.5),
        marker=dict(size=7, color=color, line=dict(color="#fff", width=1)),
        fill="toself",
        fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.13)",
    ))
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=8, color="#69f07a"), gridcolor="rgba(255,255,255,0.06)", linecolor="rgba(255,255,255,0.06)"),
            angularaxis=dict(tickfont=dict(size=9, family="Plus Jakarta Sans", color="#c8e6c9"), linecolor="rgba(255,255,255,0.06)", gridcolor="rgba(255,255,255,0.05)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(font=dict(size=10, color="#c8e6c9"), bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.08),
        margin=dict(l=40, r=40, t=20, b=20),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — RECOMMENDED INTERVENTIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header" style="margin-top:18px;">
  <div class="section-num">4</div>
  <span class="section-title">Recommended Interventions</span>
  <div class="section-rule"></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="usm-card">', unsafe_allow_html=True)

def action(num, title, desc, tag_cls, tag_label, urgency_color="#1b5e20"):
    return f"""
    <div class="action-item" style="border-left: 3px solid {urgency_color};">
      <div class="action-num">{num}</div>
      <div class="action-body">
        <div class="action-title">{title}</div>
        <div class="action-desc">{desc}</div>
      </div>
      <span class="tag {tag_cls}">{tag_label}</span>
    </div>"""

if prediction == "Dropout":
    st.markdown("""
    <div class="alert-banner danger">
      <i class="ti ti-alert-triangle alert-icon danger"></i>
      <span class="alert-text danger">Critical dropout risk detected — immediate multi-department intervention required</span>
    </div>""", unsafe_allow_html=True)

    st.markdown(action(1, "Emergency Financial Aid Review",
        "Refer immediately to the Student Affairs Office for emergency scholarship/loan assistance. Suspend academic hold pending review.",
        "tag-fin", "Urgent · Finance", "#ef5350"), unsafe_allow_html=True)

    st.markdown(action(2, "Mandatory Academic Advising Session",
        "Schedule same-week meeting with the department chair. Review failed units and build a recovery/overload plan for the next semester.",
        "tag-ac", "Urgent · Academic", "#ef5350"), unsafe_allow_html=True)

    if is_displaced:
        st.markdown(action(3, "Housing & Welfare Check",
            "Coordinate with the OSA to assess dormitory or boarding situation. Displaced students are at elevated risk of withdrawal.",
            "tag-eng", "Social", "#ff9800"), unsafe_allow_html=True)

    if has_special_needs:
        st.markdown(action(4, "Special Education Support",
            "Connect student with the Guidance & Counseling Center. Ensure IEP or accommodation plan is active and enforced by instructors.",
            "tag-sch", "Support", "#ff9800"), unsafe_allow_html=True)

    st.markdown(action(5, "30-Day Re-Assessment",
        "Flag student for follow-up review in 30 days. Document all interventions and track attendance and grade recovery.",
        "tag-ac", "Monitoring", "#78909c"), unsafe_allow_html=True)

elif prediction == "Enrolled":
    st.markdown("""
    <div class="alert-banner warning">
      <i class="ti ti-alert-circle alert-icon warning"></i>
      <span class="alert-text warning">Moderate dropout risk — preventive monitoring and proactive outreach advised</span>
    </div>""", unsafe_allow_html=True)

    st.markdown(action(1, "Mid-Semester Academic Check-In",
        "Assign an academic advisor for a structured mid-term meeting. Review current grades, attendance patterns, and subject load.",
        "tag-ac", "Academic", "#ff9800"), unsafe_allow_html=True)

    if is_debtor:
        st.markdown(action(2, "Financial Counseling",
            "Student has an outstanding balance. Coordinate a payment plan or refer to the scholarship office to prevent enrollment hold.",
            "tag-fin", "Finance", "#ff9800"), unsafe_allow_html=True)

    if sem_trend > 0.3:
        st.markdown(action(3, "Peer Tutoring Referral",
            "Declining grade trend detected between semesters. Refer to the Academic Excellence Program (AEP) for supplemental instruction.",
            "tag-ac", "Academic", "#f9a825"), unsafe_allow_html=True)

    st.markdown(action(4, "Motivational Counseling",
        "Initiate a check-in with the Guidance Office to assess student motivation, study habits, and personal challenges.",
        "tag-sch", "Support", "#78909c"), unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="alert-banner success">
      <i class="ti ti-circle-check alert-icon success"></i>
      <span class="alert-text success">Low dropout risk — student profile consistent with on-track graduates</span>
    </div>""", unsafe_allow_html=True)

    st.markdown(action("✓", "Standard Progress Monitoring",
        "No emergency intervention required. Continue routine semestral advising and tracking through the standard EWS dashboard.",
        "tag-ac", "Routine", "#1b5e20"), unsafe_allow_html=True)

    if has_scholarship:
        st.markdown(action("✓", "Scholarship Renewal Reminder",
            "Verify that the student meets the GWA and unit requirements for scholarship renewal before the next semester starts.",
            "tag-fin", "Finance", "#1b5e20"), unsafe_allow_html=True)

    st.markdown(action("✓", "Career & Graduation Planning",
        "Encourage enrollment in career development programs and pre-graduation requirements audit with the Registrar's Office.",
        "tag-sch", "Development", "#1b5e20"), unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="usm-footer">
  University of Southern Mindanao · Student Retention Early Warning System · Capstone 2026
</div>
""", unsafe_allow_html=True)