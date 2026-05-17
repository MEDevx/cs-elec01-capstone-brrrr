import base64

import streamlit as st


def configure_page() -> None:
    st.set_page_config(
        page_title="USM - Student Retention EWS",
        layout="wide",
        page_icon="assets/usm_seal.png",
        initial_sidebar_state="collapsed",
    )


def get_base64_image(path: str) -> str:
    try:
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode()
    except FileNotFoundError:
        return ""


def inject_styles() -> None:
    theme = st.session_state.get("theme", "Light")
    
    if theme == "Dark":
        # True Dark (AMOLED Style)
        colors = """
        :root {
          --g: #4ade80;        
          --g-light: #062016;
          --g-dark: #022c1b;
          --gold: #fbbf24;
          --gold-light: #2d1a01;
          
          --bg: #000000;       /* Pure Black */
          --card-bg: #0a0a0a;  /* Very Dark Grey */
          --text-main: #ffffff;
          --text-muted: #a1a1aa;
          --text-light: #52525b;
          
          --border: #18181b;   /* Subtle zinc border */
          --border-focus: #4ade80;
          
          --red: #f87171;
          --red-bg: #450a0a;
          --amber: #fbbf24;
          --amber-bg: #451a03;
          --success: #34d399;
          --success-bg: #064e3b;

          --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.5);
          --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.6);
          --shadow-md: 0 10px 15px -3px rgb(0 0 0 / 0.8);
          
          --rad: 12px;
          --rad-sm: 8px;
          --font-body: "Plus Jakarta Sans", sans-serif;
          --font-serif: "Libre Baskerville", serif;
        }
        """
    else:
        # Modern Professional Light
        colors = """
        :root {
          --g: #1b5e20;        /* USM Green Primary */
          --g-light: #e8f5e9;  /* Soft Green Background */
          --g-dark: #003300;
          --gold: #f9a825;     /* USM Gold Accent */
          --gold-light: #fffde7;
          
          --bg: #f8fafc;       /* App background */
          --card-bg: #ffffff;  /* Content card background */
          --text-main: #1e293b;
          --text-muted: #64748b;
          --text-light: #94a3b8;
          
          --border: #e2e8f0;
          --border-focus: #1b5e20;
          
          --red: #ef4444;
          --red-bg: #fef2f2;
          --amber: #f59e0b;
          --amber-bg: #fffbeb;
          --success: #10b981;
          --success-bg: #f0fdf4;

          --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
          --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
          --shadow-md: 0 10px 15px -3px rgb(0 0 0 / 0.1);

          --rad: 12px;
          --rad-sm: 8px;
          --font-body: "Plus Jakarta Sans", sans-serif;
          --font-serif: "Libre Baskerville", serif;
        }
        """

    st.markdown(f"<style>{colors}</style>", unsafe_allow_html=True)

    st.markdown(
        """
<link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@latest/tabler-icons.min.css">
""",
        unsafe_allow_html=True,
    )

    try:
        with open("style.css", encoding="utf-8") as file:
            st.markdown(f"<style>{file.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


def render_top_nav() -> None:
    logo_b64 = get_base64_image("assets/usm_seal.png")
    fallback = '<span style="font-size:20px;">USM</span>'
    logo = f'<img src="data:image/png;base64,{logo_b64}" alt="USM Seal">' if logo_b64 else fallback
    st.markdown(
        f"""
<div class="usm-nav">
  <div class="usm-logo-ring">
    {logo}
  </div>
  <div>
    <div class="usm-nav-title">University of Southern Mindanao</div>
    <div class="usm-nav-sub">Student Retention Early Warning System</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
<div class="usm-footer">
  University of Southern Mindanao · Student Retention Early Warning System · Capstone 2026
</div>
""",
        unsafe_allow_html=True,
    )
