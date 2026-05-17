from dataclasses import dataclass

import pandas as pd
import streamlit as st

from usm_ews.constants import (
    APPLICATION_MODE,
    APPLICATION_ORDER,
    BINARY,
    COURSE,
    GENDER,
    HOUSEHOLD_INCOME,
    MARITAL_STATUS,
    OCCUPATION,
    PREV_QUALIFICATION,
    QUALIFICATION_LEVEL,
)


@dataclass
class SidebarState:
    user_inputs: dict[str, float | int]
    student_name: str
    student_id: str
    raw_grade_1: float
    raw_grade_2: float
    analyze_clicked: bool


def section_header(num: str, title: str) -> None:
    st.markdown(
        f"""
<div class="section-header">
  <div class="section-num">{num}</div>
  <div class="section-title">{title}</div>
  <div class="section-rule"></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_inputs(df_raw: pd.DataFrame, feature_cols: list[str]) -> SidebarState:
    st.markdown(
        """
<div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
  <i class="ti ti-file-description" style="font-size: 24px; color: var(--g);"></i>
  <h3 style="margin:0; color: var(--text-main);">Student Data Entry</h3>
</div>
""",
        unsafe_allow_html=True,
    )
    st.info("Please fill in the details below to generate a retention risk analysis.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            ":material/person: Profile & Demographics",
            ":material/school: Academic Background",
            ":material/query_stats: 1st Sem Performance",
            ":material/trending_up: 2nd Sem Performance",
            ":material/payments: Economic Context",
        ]
    )

    user_inputs: dict[str, float | int] = {}

    def cat_input(label: str, mapping: dict[str, int | float], col_name: str, key: str):
        default_val = int(df_raw[col_name].mode()[0]) if col_name in df_raw.columns else list(mapping.values())[0]
        default_key = next((name for name, value in mapping.items() if value == default_val), list(mapping.keys())[0])
        choice = st.selectbox(label, list(mapping.keys()), index=list(mapping.keys()).index(default_key), key=key)
        return mapping[choice]

    def num_input(label: str, col_name: str, key: str, step: float = 1.0, fmt: str = "%.0f"):
        default = float(df_raw[col_name].mean()) if col_name in df_raw.columns else 0.0
        return st.number_input(label, value=round(default, 2), step=step, format=fmt, key=key)

    with tab1:
        section_header("1", "Student Identity")
        col1, col2 = st.columns(2)
        with col1:
            student_name = st.text_input("Student Name", placeholder="e.g. Juan Dela Cruz (optional)")
        with col2:
            student_id = st.text_input("Student ID", placeholder="e.g. 2024-0192 (optional)")

        section_header("2", "Demographics & Enrollment")
        col3, col4 = st.columns(2)
        with col3:
            user_inputs["Marital status"] = cat_input("Marital Status", MARITAL_STATUS, "Marital status", "marital")
            user_inputs["Gender"] = cat_input("Gender", GENDER, "Gender", "gender")
        with col4:
            user_inputs["Age at enrollment"] = int(num_input("Age at Enrollment", "Age at enrollment", "age"))
            user_inputs["Displaced"] = cat_input("Living Away from Home / Relocated", BINARY, "Displaced", "displaced")

        user_inputs["Nacionality"] = 1
        user_inputs["International"] = 0

        section_header("3", "Application Details")
        col5, col6 = st.columns(2)
        with col5:
            user_inputs["Application mode"] = cat_input("Application Mode", APPLICATION_MODE, "Application mode", "app_mode")
            user_inputs["Application order"] = cat_input("Application Order", APPLICATION_ORDER, "Application order", "app_order")
        with col6:
            user_inputs["Course"] = cat_input("Course", COURSE, "Course", "course")
            attendance_col = next((col for col in feature_cols if "attendance" in col.lower()), "Daytime/evening attendance\t")
            user_inputs[attendance_col] = 1

    with tab2:
        section_header("4", "Academic Background")
        col7, col8 = st.columns(2)
        with col7:
            user_inputs["Previous qualification"] = cat_input("Previous Qualification", PREV_QUALIFICATION, "Previous qualification", "prev_qual")
            raw_prev_grade = st.number_input(
                "Senior High School GWA (75-100)",
                min_value=75.0,
                max_value=100.0,
                value=85.0,
                step=0.5,
                format="%.1f",
                key="shs_gwa",
            )
            user_inputs["Previous qualification (grade)"] = round((raw_prev_grade - 75) / 25 * 200, 2)
        with col8:
            raw_admission = st.number_input(
                "USMCEE Score (%)",
                min_value=0.0,
                max_value=100.0,
                value=55.0,
                step=0.5,
                format="%.1f",
                key="usmcee",
            )
            user_inputs["Admission grade"] = round(raw_admission * 2, 2)

        section_header("5", "Family Background")
        col9, col10 = st.columns(2)
        with col9:
            user_inputs["Mother's qualification"] = cat_input("Mother's Qualification", QUALIFICATION_LEVEL, "Mother's qualification", "m_qual")
            user_inputs["Mother's occupation"] = cat_input("Mother's Occupation", OCCUPATION, "Mother's occupation", "m_occ")
        with col10:
            user_inputs["Father's qualification"] = cat_input("Father's Qualification", QUALIFICATION_LEVEL, "Father's qualification", "f_qual")
            user_inputs["Father's occupation"] = cat_input("Father's Occupation", OCCUPATION, "Father's occupation", "f_occ")

        section_header("6", "Support & Status")
        col11, col12 = st.columns(2)
        with col11:
            user_inputs["Educational special needs"] = cat_input("Special Educational Needs", BINARY, "Educational special needs", "sen")
            user_inputs["Debtor"] = cat_input("Is Debtor", BINARY, "Debtor", "debtor")
        with col12:
            user_inputs["Tuition fees up to date"] = cat_input("Fees Up to Date", BINARY, "Tuition fees up to date", "fees")
            user_inputs["Scholarship holder"] = cat_input("Scholarship Holder", BINARY, "Scholarship holder", "scholar")

    with tab3:
        section_header("7", "1st Semester Performance")
        c1, c2, c3 = st.columns(3)
        with c1:
            user_inputs["Curricular units 1st sem (credited)"] = int(num_input("Units Credited", "Curricular units 1st sem (credited)", "u1_cred"))
            user_inputs["Curricular units 1st sem (enrolled)"] = int(num_input("Units Enrolled", "Curricular units 1st sem (enrolled)", "u1_enr"))
        with c2:
            user_inputs["Curricular units 1st sem (evaluations)"] = int(num_input("Units Evaluated", "Curricular units 1st sem (evaluations)", "u1_eval"))
            user_inputs["Curricular units 1st sem (approved)"] = int(num_input("Units Approved", "Curricular units 1st sem (approved)", "u1_app"))
        with c3:
            raw_grade_1 = st.number_input(
                "Grade Average (1.0 - 5.0)",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.1,
                format="%.2f",
                key="g1_avg",
            )
            user_inputs["Curricular units 1st sem (grade)"] = round((5.0 - raw_grade_1) / 4.0 * 20, 2)
            user_inputs["Curricular units 1st sem (without evaluations)"] = int(num_input("Without Eval", "Curricular units 1st sem (without evaluations)", "u1_noeval"))

    with tab4:
        section_header("8", "2nd Semester Performance")
        c4, c5, c6 = st.columns(3)
        with c4:
            user_inputs["Curricular units 2nd sem (credited)"] = int(num_input("Units Credited", "Curricular units 2nd sem (credited)", "u2_cred"))
            user_inputs["Curricular units 2nd sem (enrolled)"] = int(num_input("Units Enrolled", "Curricular units 2nd sem (enrolled)", "u2_enr"))
        with c5:
            user_inputs["Curricular units 2nd sem (evaluations)"] = int(num_input("Units Evaluated", "Curricular units 2nd sem (evaluations)", "u2_eval"))
            user_inputs["Curricular units 2nd sem (approved)"] = int(num_input("Units Approved", "Curricular units 2nd sem (approved)", "u2_app"))
        with c6:
            raw_grade_2 = st.number_input(
                "Grade Average (1.0 - 5.0)",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.1,
                format="%.2f",
                key="g2_avg",
            )
            user_inputs["Curricular units 2nd sem (grade)"] = round((5.0 - raw_grade_2) / 4.0 * 20, 2)
            user_inputs["Curricular units 2nd sem (without evaluations)"] = int(num_input("Without Eval", "Curricular units 2nd sem (without evaluations)", "u2_noeval"))

    with tab5:
        section_header("9", "Economic Context")
        col13, col14 = st.columns(2)
        with col13:
            user_inputs["Unemployment rate"] = num_input("Unemployment Rate (%)", "Unemployment rate", "unemp", step=0.1, fmt="%.1f")
            user_inputs["Inflation rate"] = num_input("Inflation Rate (%)", "Inflation rate", "infl", step=0.1, fmt="%.1f")
        with col14:
            income_label = st.selectbox("Estimated Household Income", list(HOUSEHOLD_INCOME.keys()), key="income")
            user_inputs["GDP"] = HOUSEHOLD_INCOME[income_label]

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    col_run, col_reset = st.columns([0.82, 0.18])
    with col_run:
        analyze_clicked = st.button("Run Risk Analysis", icon=":material/analytics:", use_container_width=True, type="primary")
    with col_reset:
        if st.button("Reset", icon=":material/refresh:", use_container_width=True):
            st.rerun()

    return SidebarState(
        user_inputs=user_inputs,
        student_name=student_name,
        student_id=student_id,
        raw_grade_1=raw_grade_1,
        raw_grade_2=raw_grade_2,
        analyze_clicked=analyze_clicked,
    )
