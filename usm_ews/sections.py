import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from usm_ews.constants import COURSE


def render_prediction_verdict(state, model_accuracy: float, feature_cols: list[str], df_raw: pd.DataFrame) -> None:
    st.markdown(
        """
<div class="section-header">
  <div class="section-num">1</div>
  <span class="section-title">Classification Result</span>
  <div class="section-rule"></div>
</div>
""",
        unsafe_allow_html=True,
    )

    acc_display = round(model_accuracy * 100, 1)

    if state.prediction == "Dropout":
        risk_level, risk_cls, risk_icon = "HIGH RISK", "danger", "ti-alert-triangle"
    elif state.prediction == "Enrolled":
        risk_level, risk_cls, risk_icon = "MODERATE RISK", "warning", "ti-alert-circle"
    else:
        risk_level, risk_cls, risk_icon = "LOW RISK", "success", "ti-circle-check"

    conf_ring_pct = round(state.confidence)
    student_name = st.session_state.get("student_name_value", "")
    student_id = st.session_state.get("student_id_value", "")

    name_html = (
        f'<strong style="color:var(--text-main); font-size:15px;">{student_name}</strong>'
        if student_name
        else '<span style="color:var(--text-light); font-style:italic;">No name provided</span>'
    )
    id_html = (
        f'&nbsp;<span style="color:var(--border);">·</span>&nbsp; '
        f'<span style="color:var(--text-muted);">ID: {student_id}</span>'
        if student_id
        else ""
    )

    st.markdown(
        f"""
<div class="verdict-hero" style="border-left: 6px solid {state.color};">
  <div class="verdict-left">
    <div class="verdict-tag {risk_cls}">
      <i class="ti {risk_icon}"></i> {risk_level}
    </div>
    <div class="verdict-outcome" style="color:{state.color};">{state.prediction}</div>
    <div class="verdict-sub">
      {name_html}{id_html}
      &nbsp;<span style="color:var(--border);">·</span>&nbsp; {len(feature_cols)} features analysed
    </div>
    <div class="verdict-meta-row">
      <div class="verdict-meta-pill"><i class="ti ti-binary-tree-2"></i> Random Forest Model</div>
      <div class="verdict-meta-pill"><i class="ti ti-crosshair"></i> Accuracy: {acc_display}%</div>
      <div class="verdict-meta-pill"><i class="ti ti-database"></i> n = {len(df_raw)}</div>
    </div>
  </div>
  <div class="verdict-right">
    <div class="conf-donut-wrap">
      <svg width="110" height="110" viewBox="0 0 110 110">
        <circle cx="55" cy="55" r="46" fill="none" stroke="var(--bg)" stroke-width="10"/>
        <circle cx="55" cy="55" r="46" fill="none" stroke="{state.color}" stroke-width="10"
          stroke-dasharray="{round(conf_ring_pct * 2.89)} 289"
          stroke-dashoffset="72" stroke-linecap="round" transform="rotate(-90 55 55)"/>
      </svg>
      <div class="conf-donut-label">
        <div class="conf-pct" style="color:{state.color};">{conf_ring_pct}%</div>
        <div class="conf-caption">confidence</div>
      </div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="prob-row-wrap" style="background:var(--bg); border:1px solid var(--border); padding:20px; border-radius:12px; margin-bottom:24px;">
  <div class="prob-row-title" style="font-size:12px; font-weight:700; color:var(--text-muted); text-transform:uppercase; margin-bottom:15px; letter-spacing:1px;">Class Probability Breakdown</div>
  <div class="prob-row" style="display:flex; gap:15px;">
""",
        unsafe_allow_html=True,
    )

    probs = [
        ("Dropout", state.dropout_prob, "#ef4444"),
        ("Enrolled", state.enrolled_prob, "#f59e0b"),
        ("Graduate", state.graduate_prob, "#10b981"),
    ]
    for label, prob, color in probs:
        active_border = f"border: 2px solid {color};" if label == state.prediction else "border: 1px solid var(--border);"
        active_bg = f"background: {color}08;" if label == state.prediction else "background: white;"
        st.markdown(
            f"""
    <div class="prob-col" style="flex:1; padding:15px; border-radius:10px; {active_border} {active_bg}">
      <div class="prob-label" style="font-size:11px; font-weight:700; color:var(--text-muted); margin-bottom:8px; text-transform:uppercase;">{label}</div>
      <div class="prob-bar-track" style="height:6px; background:var(--bg); border-radius:3px; margin-bottom:8px; overflow:hidden;">
        <div class="prob-bar-fill" style="width:{prob:.1f}%; height:100%; background:{color}; border-radius:3px;"></div>
      </div>
      <div class="prob-pct" style="font-size:20px; font-weight:700; color:{color}; font-family:var(--font-serif);">{prob:.1f}%</div>
    </div>
    """,
            unsafe_allow_html=True,
        )

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_risk_snapshot(state) -> None:
    st.markdown(
        """
<div class="section-header" style="margin-top:18px;">
  <div class="section-num">2</div>
  <span class="section-title">Student Risk Snapshot</span>
  <div class="section-rule"></div>
</div>
""",
        unsafe_allow_html=True,
    )

    kpi1_color = "#ef4444" if state.pass_rate_1 < 60 else ("#f59e0b" if state.pass_rate_1 < 80 else "#10b981")
    kpi2_color = "#ef4444" if state.pass_rate_2 < 60 else ("#f59e0b" if state.pass_rate_2 < 80 else "#10b981")
    trend_icon = "ti-trending-down" if state.sem_trend > 0 else "ti-trending-up"
    trend_color = "#ef4444" if state.sem_trend > 0 else "#10b981"
    trend_label = f"+{state.sem_trend:.2f} (declining)" if state.sem_trend > 0 else f"{state.sem_trend:.2f} (improving)"
    fin_status = "Debtor" if state.is_debtor else ("Scholar" if state.has_scholarship else "Self-Funded")
    fin_color = "#ef4444" if state.is_debtor else ("#f59e0b" if state.has_scholarship else "#64748b")

    st.markdown(
        f"""
<div class="kpi-strip">
  <div class="kpi-tile">
    <div class="kpi-icon-wrap" style="background:var(--g-light);"><i class="ti ti-school" style="color:var(--g);"></i></div>
    <div class="kpi-body">
      <div class="kpi-val" style="color:{kpi1_color};">{state.pass_rate_1}%</div>
      <div class="kpi-lbl">1st Sem Pass Rate</div>
      <div class="kpi-sub">{state.approved_1} of {state.enrolled_1} units</div>
    </div>
  </div>
  <div class="kpi-tile">
    <div class="kpi-icon-wrap" style="background:var(--g-light);"><i class="ti ti-school" style="color:var(--g);"></i></div>
    <div class="kpi-body">
      <div class="kpi-val" style="color:{kpi2_color};">{state.pass_rate_2}%</div>
      <div class="kpi-lbl">2nd Sem Pass Rate</div>
      <div class="kpi-sub">{state.approved_2} of {state.enrolled_2} units</div>
    </div>
  </div>
  <div class="kpi-tile">
    <div class="kpi-icon-wrap" style="background:var(--amber-bg);"><i class="ti {trend_icon}" style="color:{trend_color};"></i></div>
    <div class="kpi-body">
      <div class="kpi-val" style="color:{trend_color};">{state.grade_2_raw:.2f}</div>
      <div class="kpi-lbl">Grade Trend (2nd Sem)</div>
      <div class="kpi-sub">{trend_label}</div>
    </div>
  </div>
  <div class="kpi-tile">
    <div class="kpi-icon-wrap" style="background:var(--bg);"><i class="ti ti-wallet" style="color:{fin_color};"></i></div>
    <div class="kpi-body">
      <div class="kpi-val" style="color:{fin_color};">{fin_status}</div>
      <div class="kpi-lbl">Financial Status</div>
      <div class="kpi-sub">{'Unpaid balance' if state.is_debtor else ('Grant active' if state.has_scholarship else 'No aid')}</div>
    </div>
  </div>
  <div class="kpi-tile">
    <div class="kpi-icon-wrap" style="background:var(--red-bg);"><i class="ti ti-flag-3" style="color:#ef4444;"></i></div>
    <div class="kpi-body">
      <div class="kpi-val" style="color:#ef4444;">{len(state.risk_flags)}</div>
      <div class="kpi-lbl">Risk Flags</div>
      <div class="kpi-sub">{'Immediate action' if len(state.risk_flags) >= 3 else ('Monitor' if len(state.risk_flags) > 0 else 'None')}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    if state.risk_flags:
        pills_html = (
            '<div class="flag-pills-wrap" style="background:var(--bg); padding:15px; border-radius:12px; border:1px solid var(--border);">'
            '<div class="flag-pills-label" style="font-size:12px; font-weight:700; color:var(--text-muted); margin-bottom:12px; display:flex; align-items:center; gap:8px;">'
            '<i class="ti ti-flag-3" style="color:var(--red);"></i> Active Risk Flags</div><div class="flag-pills" style="display:flex; flex-wrap:wrap; gap:8px;">'
        )
        flag_tag_map = {
            "academic": "tag-ac",
            "financial": "tag-fin",
            "social": "tag-eng",
            "support": "tag-sch",
        }
        for flag_name, flag_type in state.risk_flags:
            pills_html += f'<span class="tag {flag_tag_map.get(flag_type, "tag-ac")}">{flag_name}</span>'
        pills_html += "</div></div>"
        st.markdown(pills_html, unsafe_allow_html=True)


def render_model_analytics(state, model, feature_cols: list[str], user_inputs: dict[str, int | float]) -> None:
    st.markdown(
        """
<div class="section-header" style="margin-top:18px;">
  <div class="section-num">3</div>
  <span class="section-title">Model Analytics</span>
  <div class="section-rule"></div>
</div>
""",
        unsafe_allow_html=True,
    )

    c3, c4 = st.columns([1.6, 1.4], gap="large")

    with c3:
        st.markdown('<div class="usm-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><i class="ti ti-chart-bar"></i> Top Feature Importances</div>', unsafe_allow_html=True)

        imps = model.feature_importances_
        top_indices = np.argsort(imps)[-7:]
        top_imps = imps[top_indices]
        raw_feats = np.array(feature_cols)[top_indices]
        clean_feats = [str(feat).replace("Curricular units", "CU").replace(" attendance\t", "").strip() for feat in raw_feats]

        bar_colors = []
        for feat in raw_feats:
            lowered = str(feat).lower()
            if "grade" in lowered or "approved" in lowered or "enrolled" in lowered or "curricular" in lowered:
                bar_colors.append("#10b981")
            elif "gdp" in lowered or "unemploy" in lowered or "inflation" in lowered or "tuition" in lowered or "debtor" in lowered:
                bar_colors.append("#f59e0b")
            else:
                bar_colors.append("#3b82f6")

        fi_fig = go.Figure(
            go.Bar(
                x=top_imps * 100,
                y=clean_feats,
                orientation="h",
                marker_color=bar_colors,
                marker_line_width=0,
                marker_cornerradius=4,
                text=[f"{round(val * 100, 1)}%" for val in top_imps],
                textposition="outside",
                textfont=dict(size=11, color="var(--text-main)"),
            )
        )
        fi_fig.update_layout(
            xaxis=dict(ticksuffix="%", tickfont=dict(size=10, color="var(--text-muted)"), gridcolor="var(--border)", showgrid=True),
            yaxis=dict(tickfont=dict(size=11, color="var(--text-main)"), showgrid=False, automargin=True),
            height=300,
            margin=dict(l=10, r=50, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            bargap=0.3,
        )
        st.plotly_chart(fi_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            """
    <div style="display:flex; gap:14px; margin-top:10px; padding: 0 4px;">
      <span style="font-size:11px; color:var(--text-muted);"><span style="color:#10b981;">●</span> Academic</span>
      <span style="font-size:11px; color:var(--text-muted);"><span style="color:#f59e0b;">●</span> Economic</span>
      <span style="font-size:11px; color:var(--text-muted);"><span style="color:#3b82f6;">●</span> Demographic</span>
    </div>
    """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="usm-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><i class="ti ti-chart-line"></i> Semester Comparison</div>', unsafe_allow_html=True)

        sem_labels = ["1st Sem", "2nd Sem"]
        pass_vals = [state.pass_rate_1, state.pass_rate_2]
        grade_vals = [state.grade_1_raw, state.grade_2_raw]

        sem_fig = go.Figure()
        sem_fig.add_trace(
            go.Bar(
                name="Pass Rate (%)",
                x=sem_labels,
                y=pass_vals,
                marker_color=["#10b981" if val >= 75 else "#ef4444" for val in pass_vals],
                marker_cornerradius=5,
                yaxis="y1",
                offsetgroup=1,
                text=[f"{val}%" for val in pass_vals],
                textposition="outside",
                textfont=dict(size=12, color="var(--text-main)"),
            )
        )
        sem_fig.add_trace(
            go.Scatter(
                name="Grade Avg",
                x=sem_labels,
                y=grade_vals,
                mode="lines+markers+text",
                line=dict(color="var(--gold)", width=3, dash="dot"),
                marker=dict(size=10, color="var(--gold)", line=dict(color="#fff", width=2)),
                yaxis="y2",
                text=[f"{val:.2f}" for val in grade_vals],
                textposition="top center",
                textfont=dict(size=12, color="var(--gold)"),
            )
        )
        sem_fig.update_layout(
            yaxis=dict(title="Pass Rate (%)", range=[0, 120], tickfont=dict(size=10, color="var(--text-muted)"), gridcolor="var(--border)", showgrid=True, ticksuffix="%"),
            yaxis2=dict(title="Grade (1.0-5.0)", overlaying="y", side="right", range=[0.5, 6], tickfont=dict(size=10, color="var(--gold)"), showgrid=False),
            xaxis=dict(tickfont=dict(size=12, color="var(--text-main)"), showgrid=False),
            legend=dict(font=dict(size=11, color="var(--text-muted)"), bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.2),
            height=300,
            margin=dict(l=10, r=40, t=10, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="group",
            bargap=0.4,
        )
        st.plotly_chart(sem_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    c5, c6 = st.columns([1, 2], gap="large")

    with c5:
        age_val = user_inputs.get("Age at enrollment", "-")
        gender_val = "Male" if user_inputs.get("Gender", 0) == 1 else "Female"
        scholar_val = "Yes" if state.has_scholarship else "No"
        debtor_val = "Yes" if state.is_debtor else "No"
        course_val = next((name for name, value in COURSE.items() if value == user_inputs.get("Course")), "-")
        course_short = f"{course_val[:28]}..." if len(course_val) > 28 else course_val
        displaced_val = "Yes" if state.is_displaced else "No"
        report_date = pd.Timestamp.now().strftime("%b %d, %Y")
        student_name = st.session_state.get("student_name_value", "")
        student_id = st.session_state.get("student_id_value", "")

        st.markdown(
            f"""
    <div class="usm-card" style="height:100%;">
      <div class="card-title"><i class="ti ti-id-badge-2"></i> Student Summary</div>
      <div class="summary-row">
        <span class="sum-lbl">Name</span>
        <span class="sum-val">{student_name if student_name else '<em style="color:var(--text-light);">-</em>'}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Student ID</span>
        <span class="sum-val">{student_id if student_id else '<em style="color:var(--text-light);">-</em>'}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Age / Gender</span>
        <span class="sum-val">{age_val} · {gender_val}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Program</span>
        <span class="sum-val" title="{course_val}">{course_short}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Scholarship</span>
        <span class="sum-val" style="color:{'#10b981' if scholar_val == 'Yes' else 'var(--text-muted)'};">{scholar_val}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Debtor</span>
        <span class="sum-val" style="color:{'#ef4444' if debtor_val == 'Yes' else 'var(--text-muted)'};">{debtor_val}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Displaced</span>
        <span class="sum-val" style="color:{'#f59e0b' if displaced_val == 'Yes' else 'var(--text-muted)'};">{displaced_val}</span>
      </div>
      <div class="summary-row">
        <span class="sum-lbl">Report Date</span>
        <span class="sum-val">{report_date}</span>
      </div>
    </div>
    """,
            unsafe_allow_html=True,
        )

    with c6:
        st.markdown('<div class="usm-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title"><i class="ti ti-radar"></i> Risk Factor Radar</div>', unsafe_allow_html=True)

        radar_categories = [
            "Academic\nPerformance",
            "Financial\nStability",
            "Social\nFactors",
            "Enrollment\nStrength",
            "Progression\nRate",
        ]
        acad_score = min(100, state.avg_pass_rate)
        fin_score = 80 if state.has_scholarship else (20 if state.is_debtor else 55)
        social_score = 60 if state.is_displaced else (50 if state.has_special_needs else 80)
        enroll_score = min(100, round((state.enrolled_1 + state.enrolled_2) / 2 * 5))
        prog_score = min(100, round(((state.approved_1 + state.approved_2) / max(state.enrolled_1 + state.enrolled_2, 1)) * 100))

        radar_vals = [acad_score, fin_score, social_score, enroll_score, prog_score]
        benchmark = [72, 60, 72, 65, 70]

        radar_fig = go.Figure()
        radar_fig.add_trace(
            go.Scatterpolar(
                r=benchmark + [benchmark[0]],
                theta=radar_categories + [radar_categories[0]],
                mode="lines",
                name="Average",
                line=dict(color="var(--border)", width=1.5, dash="dot"),
                fill="toself",
                fillcolor="var(--bg)",
            )
        )
        radar_fig.add_trace(
            go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_categories + [radar_categories[0]],
                mode="lines+markers",
                name="Current Student",
                line=dict(color=state.color, width=3),
                marker=dict(size=8, color=state.color, line=dict(color="#fff", width=1.5)),
                fill="toself",
                fillcolor=f"rgba({int(state.color[1:3],16)},{int(state.color[3:5],16)},{int(state.color[5:7],16)},0.15)",
            )
        )
        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=8, color="var(--text-muted)"), gridcolor="var(--border)", linecolor="var(--border)"),
                angularaxis=dict(tickfont=dict(size=10, family="Plus Jakarta Sans", color="var(--text-main)"), linecolor="var(--border)", gridcolor="var(--border)"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=True,
            legend=dict(font=dict(size=11, color="var(--text-muted)"), bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.1),
            margin=dict(l=45, r=45, t=30, b=30),
            height=260,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(radar_fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)


def render_interventions(state) -> None:
    st.markdown(
        """
<div class="section-header" style="margin-top:18px;">
  <div class="section-num">4</div>
  <span class="section-title">Recommended Interventions</span>
  <div class="section-rule"></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="usm-card">', unsafe_allow_html=True)

    def action(num, title, desc, tag_cls, tag_label, urgency_color="var(--g)"):
        return f"""
    <div class="action-item" style="border-left: 5px solid {urgency_color};">
      <div class="action-num" style="background:{urgency_color};">{num}</div>
      <div class="action-body">
        <div class="action-title">{title}</div>
        <div class="action-desc">{desc}</div>
      </div>
      <span class="tag {tag_cls}">{tag_label}</span>
    </div>"""

    if state.prediction == "Dropout":
        banner_class = "danger"
        banner_icon = "ti-alert-triangle"
        banner_text = "Critical dropout risk detected - immediate multi-department intervention required"
    elif state.prediction == "Enrolled":
        banner_class = "warning"
        banner_icon = "ti-alert-circle"
        banner_text = "Moderate dropout risk - preventive monitoring and proactive outreach advised"
    else:
        banner_class = "success"
        banner_icon = "ti-circle-check"
        banner_text = "Low dropout risk - recommendations focus on maintaining progress and smaller issues found in the analysis"

    st.markdown(
        f"""
    <div class="alert-banner {banner_class}">
      <i class="ti {banner_icon} alert-icon {banner_class}"></i>
      <span class="alert-text {banner_class}">{banner_text}</span>
    </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    <div style="margin: 0 0 14px; color: var(--text-muted); font-size: 13px;">
      These recommendations are generated from the student's analyzed inputs, including semester performance, financial status, support indicators, and risk signals.
    </div>
    """,
        unsafe_allow_html=True,
    )

    for index, recommendation in enumerate(state.interventions, start=1):
        st.markdown(
            action(
                index,
                recommendation.title,
                recommendation.desc,
                recommendation.tag_cls,
                recommendation.tag_label,
                recommendation.urgency_color,
            ),
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
