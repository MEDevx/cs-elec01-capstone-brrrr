from dataclasses import dataclass

import numpy as np
import streamlit as st

from usm_ews.constants import COLOR_MAP, LABEL_DECODER


@dataclass
class InterventionRecommendation:
    title: str
    desc: str
    tag_cls: str
    tag_label: str
    urgency_color: str


@dataclass
class PredictionState:
    prediction: str
    probabilities: np.ndarray
    confidence: float
    dropout_prob: float
    enrolled_prob: float
    graduate_prob: float
    enrolled_1: int
    approved_1: int
    enrolled_2: int
    approved_2: int
    pass_rate_1: int
    pass_rate_2: int
    avg_pass_rate: int
    grade_1_raw: float
    grade_2_raw: float
    sem_trend: float
    is_debtor: bool
    has_scholarship: bool
    is_displaced: bool
    has_special_needs: bool
    risk_flags: list[tuple[str, str]]
    interventions: list[InterventionRecommendation]
    color: str


def _build_interventions(
    prediction: str,
    user_inputs: dict[str, int | float],
    dropout_prob: float,
    pass_rate_1: int,
    pass_rate_2: int,
    avg_pass_rate: int,
    sem_trend: float,
    raw_grade_1: float,
    raw_grade_2: float,
    is_debtor: bool,
    has_scholarship: bool,
    is_displaced: bool,
    has_special_needs: bool,
) -> list[InterventionRecommendation]:
    interventions: list[InterventionRecommendation] = []

    enrolled_1 = int(user_inputs.get("Curricular units 1st sem (enrolled)", 0))
    evaluated_1 = int(user_inputs.get("Curricular units 1st sem (evaluations)", 0))
    approved_1 = int(user_inputs.get("Curricular units 1st sem (approved)", 0))
    without_eval_1 = int(user_inputs.get("Curricular units 1st sem (without evaluations)", 0))
    enrolled_2 = int(user_inputs.get("Curricular units 2nd sem (enrolled)", 0))
    evaluated_2 = int(user_inputs.get("Curricular units 2nd sem (evaluations)", 0))
    approved_2 = int(user_inputs.get("Curricular units 2nd sem (approved)", 0))
    without_eval_2 = int(user_inputs.get("Curricular units 2nd sem (without evaluations)", 0))
    low_income = float(user_inputs.get("GDP", 0)) <= 1.2
    admission_grade = float(user_inputs.get("Admission grade", 0)) / 2
    shs_gwa = 75 + (float(user_inputs.get("Previous qualification (grade)", 0)) / 200 * 25)

    def add(title: str, desc: str, tag_cls: str, tag_label: str, urgency_color: str) -> None:
        interventions.append(
            InterventionRecommendation(
                title=title,
                desc=desc,
                tag_cls=tag_cls,
                tag_label=tag_label,
                urgency_color=urgency_color,
            )
        )

    if avg_pass_rate < 60 or dropout_prob >= 45:
        add(
            "Intensive Academic Recovery Plan",
            (
                f"Average pass rate is {avg_pass_rate}% with a {dropout_prob:.1f}% dropout probability. "
                "Schedule an adviser meeting this week to review failed units, adjust load, and assign subject-specific tutorials."
            ),
            "tag-ac",
            "Priority · Academic",
            "#ef4444",
        )

    if pass_rate_1 < 75 or pass_rate_2 < 75 or sem_trend > 0.30:
        add(
            "Tutoring and Study Support Referral",
            (
                f"Semester grades moved from {raw_grade_1:.2f} to {raw_grade_2:.2f}. "
                "Refer the student to peer tutoring or remedial coaching for courses with low completion or declining performance."
            ),
            "tag-ac",
            "Academic",
            "#f59e0b",
        )

    if without_eval_1 > 0 or without_eval_2 > 0 or evaluated_1 < enrolled_1 or evaluated_2 < enrolled_2:
        add(
            "Attendance and Class Engagement Follow-Up",
            (
                f"The record shows {without_eval_1 + without_eval_2} unit(s) without evaluation. "
                "Check attendance, missing requirements, and faculty feedback before the issue turns into non-completion."
            ),
            "tag-eng",
            "Monitoring",
            "#f59e0b",
        )

    if is_debtor or (low_income and not has_scholarship):
        finance_note = "The student has an unpaid balance on record." if is_debtor else "The income profile suggests the student may need financial support."
        add(
            "Financial Aid and Payment Counseling",
            (
                f"{finance_note} Review scholarship eligibility, installment options, and emergency assistance to reduce withdrawal pressure."
            ),
            "tag-fin",
            "Priority · Finance",
            "#ef4444" if is_debtor else "#f59e0b",
        )

    if not has_scholarship and (avg_pass_rate >= 75 or not is_debtor):
        add(
            "Scholarship Screening",
            (
                "The student is currently not tagged as a scholarship holder. "
                "Assess eligibility for grants, vouchers, or work-study support that can strengthen retention."
            ),
            "tag-fin",
            "Finance",
            "#f59e0b",
        )

    if is_displaced:
        add(
            "Housing and Adjustment Support",
            (
                "The student is marked as displaced or relocated for study. "
                "Coordinate a welfare check on housing, transport, and daily adjustment needs."
            ),
            "tag-eng",
            "Social",
            "#f59e0b",
        )

    if has_special_needs:
        add(
            "Accessibility and Counseling Coordination",
            (
                "The analysis indicates special educational needs. "
                "Confirm accommodations, connect with guidance services, and check whether classroom support is active."
            ),
            "tag-sch",
            "Support",
            "#8b5cf6",
        )

    if admission_grade < 55 or shs_gwa < 82:
        add(
            "Foundation Skills Bridging",
            (
                f"Entry indicators are modest (USMCEE {admission_grade:.1f}%, SHS GWA {shs_gwa:.1f}). "
                "Recommend study-skills coaching, diagnostic review, and bridge support in core subjects."
            ),
            "tag-ac",
            "Preparedness",
            "#64748b",
        )

    if prediction == "Graduate" and avg_pass_rate >= 75 and sem_trend <= 0.0:
        add(
            "Progression and Career Planning",
            (
                "Current results are consistent with an on-track student profile. "
                "Continue regular advising and start internship, graduation, or career preparation planning."
            ),
            "tag-sch",
            "Development",
            "#10b981",
        )

    if not interventions:
        add(
            "Routine Adviser Check-In",
            (
                "No major intervention trigger was detected from the current analysis. "
                "Keep the student on standard monitoring and review the profile again next term."
            ),
            "tag-ac",
            "Routine",
            "#10b981",
        )

    return interventions[:6]


def build_prediction_state(model, feature_cols: list[str], user_inputs: dict[str, int | float], raw_grade_1: float, raw_grade_2: float):
    try:
        input_data = np.array([[user_inputs[col] for col in feature_cols]])
    except KeyError as exc:
        st.error(f"Feature mismatch error. Expected column not found in inputs: {exc}")
        st.stop()

    raw_pred = model.predict(input_data)[0]
    prediction = LABEL_DECODER.get(raw_pred, "Unknown")
    probabilities = model.predict_proba(input_data)[0]
    confidence = max(probabilities) * 100 if len(probabilities) > 0 else 0

    dropout_prob = probabilities[0] * 100 if len(probabilities) > 0 else 0
    enrolled_prob = probabilities[1] * 100 if len(probabilities) > 1 else 0
    graduate_prob = probabilities[2] * 100 if len(probabilities) > 2 else 0

    enrolled_1 = int(user_inputs.get("Curricular units 1st sem (enrolled)", 1))
    approved_1 = int(user_inputs.get("Curricular units 1st sem (approved)", 0))
    enrolled_2 = int(user_inputs.get("Curricular units 2nd sem (enrolled)", 1))
    approved_2 = int(user_inputs.get("Curricular units 2nd sem (approved)", 0))
    pass_rate_1 = round((approved_1 / enrolled_1) * 100) if enrolled_1 > 0 else 0
    pass_rate_2 = round((approved_2 / enrolled_2) * 100) if enrolled_2 > 0 else 0
    avg_pass_rate = round((pass_rate_1 + pass_rate_2) / 2)
    sem_trend = raw_grade_2 - raw_grade_1

    is_debtor = user_inputs.get("Debtor", 0) == 1
    has_scholarship = user_inputs.get("Scholarship holder", 0) == 1
    is_displaced = user_inputs.get("Displaced", 0) == 1
    has_special_needs = user_inputs.get("Educational special needs", 0) == 1

    risk_flags = []
    if dropout_prob > 40:
        risk_flags.append(("Low Approval Rate", "academic"))
    if is_debtor:
        risk_flags.append(("Financial Debt", "financial"))
    if sem_trend > 0.5:
        risk_flags.append(("Declining Grades", "academic"))
    if pass_rate_1 < 60 or pass_rate_2 < 60:
        risk_flags.append(("Failed Units", "academic"))
    if is_displaced:
        risk_flags.append(("Relocated for Study", "social"))
    if has_special_needs:
        risk_flags.append(("Special Ed. Needs", "support"))
    if not has_scholarship and dropout_prob > 30:
        risk_flags.append(("No Financial Aid", "financial"))

    interventions = _build_interventions(
        prediction=prediction,
        user_inputs=user_inputs,
        dropout_prob=dropout_prob,
        pass_rate_1=pass_rate_1,
        pass_rate_2=pass_rate_2,
        avg_pass_rate=avg_pass_rate,
        sem_trend=sem_trend,
        raw_grade_1=raw_grade_1,
        raw_grade_2=raw_grade_2,
        is_debtor=is_debtor,
        has_scholarship=has_scholarship,
        is_displaced=is_displaced,
        has_special_needs=has_special_needs,
    )

    return PredictionState(
        prediction=prediction,
        probabilities=probabilities,
        confidence=confidence,
        dropout_prob=dropout_prob,
        enrolled_prob=enrolled_prob,
        graduate_prob=graduate_prob,
        enrolled_1=enrolled_1,
        approved_1=approved_1,
        enrolled_2=enrolled_2,
        approved_2=approved_2,
        pass_rate_1=pass_rate_1,
        pass_rate_2=pass_rate_2,
        avg_pass_rate=avg_pass_rate,
        grade_1_raw=raw_grade_1,
        grade_2_raw=raw_grade_2,
        sem_trend=sem_trend,
        is_debtor=is_debtor,
        has_scholarship=has_scholarship,
        is_displaced=is_displaced,
        has_special_needs=has_special_needs,
        risk_flags=risk_flags,
        interventions=interventions,
        color=COLOR_MAP[prediction],
    )
