import streamlit as st

from usm_ews.data import prepare_and_train
from usm_ews.layout import configure_page, inject_styles, render_footer, render_top_nav
from usm_ews.prediction import build_prediction_state
from usm_ews.sections import (
    render_interventions,
    render_model_analytics,
    render_prediction_verdict,
    render_risk_snapshot,
)
from usm_ews.sidebar import render_inputs


def main() -> None:
    if "theme" not in st.session_state:
        st.session_state["theme"] = "Light"

    configure_page()
    inject_styles()

    artifacts = prepare_and_train()

    st.markdown('<div id="theme-toggle-marker"></div>', unsafe_allow_html=True)
    theme_icon = ":material/light_mode:" if st.session_state["theme"] == "Dark" else ":material/dark_mode:"
    if st.button(label="", icon=theme_icon, key="theme_toggle_btn", help="Switch Theme"):
        st.session_state["theme"] = "Light" if st.session_state["theme"] == "Dark" else "Dark"
        st.rerun()

    render_top_nav()

    if "analysis_run" not in st.session_state:
        st.session_state["analysis_run"] = False
    if "input_state" not in st.session_state:
        st.session_state["input_state"] = None

    if not st.session_state["analysis_run"]:
        input_state = render_inputs(artifacts.df_raw, artifacts.feature_cols)

        if input_state.analyze_clicked:
            st.session_state["analysis_run"] = True
            st.session_state["input_state"] = input_state
            st.rerun()
    else:
        input_state = st.session_state["input_state"]

        st.session_state["student_name_value"] = input_state.student_name
        st.session_state["student_id_value"] = input_state.student_id

        prediction_state = build_prediction_state(
            artifacts.model,
            artifacts.feature_cols,
            input_state.user_inputs,
            input_state.raw_grade_1,
            input_state.raw_grade_2,
        )

        col_title, col_back = st.columns([0.8, 0.2])
        with col_title:
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; gap:10px; margin: 6px 0 14px;">
                    <i class="ti ti-chart-infographic" style="font-size: 26px; color: var(--g);"></i>
                    <h2 style="margin:0; color: var(--text-main);">Analysis Results: {input_state.student_name or 'New Student'}</h2>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_back:
            if st.button("Edit Inputs", icon=":material/edit:", use_container_width=True):
                st.session_state["analysis_run"] = False
                st.rerun()

        render_prediction_verdict(
            prediction_state,
            artifacts.accuracy,
            artifacts.feature_cols,
            artifacts.df_raw,
        )
        render_risk_snapshot(prediction_state)
        render_model_analytics(
            prediction_state,
            artifacts.model,
            artifacts.feature_cols,
            input_state.user_inputs,
        )
        render_interventions(prediction_state)

    render_footer()


if __name__ == "__main__":
    main()
