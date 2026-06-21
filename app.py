import streamlit as st

from app_pages.auth_page import (
    initialise_auth_state,
    logout_user,
    require_verified_user,
    show_auth_page,
)
from app_pages.conclusion import show_conclusion
from app_pages.dataset_overview import show_dataset_overview
from app_pages.eda import show_eda
from app_pages.home import show_home
from app_pages.pyspark_kmeans import show_pyspark_kmeans
from app_pages.random_forest import show_random_forest
from utils.data_loader import load_data


st.set_page_config(
    page_title="Customer Financial Behaviour",
    layout="wide",
)


def get_chart_theme():
    """
    Reads Streamlit's top-right Light/Dark setting and sends the matching
    theme name to your Matplotlib chart pages.
    """
    try:
        return (
            "Dark Mode"
            if st.context.theme.type == "dark"
            else "Light Mode"
        )
    except AttributeError:
        # Safe fallback if an older Streamlit version is installed.
        return "Light Mode"


# One theme switch only: Streamlit's own top-right Settings theme.
theme = get_chart_theme()

# Neutral glass styling that works in both Streamlit Light and Dark themes.
st.markdown(
    """
    <style>
        .glass-card {
            background: rgba(127, 127, 127, 0.12);
            border: 1px solid rgba(127, 127, 127, 0.28);
            border-radius: 28px;
            padding: 28px;
            margin-bottom: 24px;
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
        }

        div[data-testid="stMetric"] {
            border-radius: 20px;
            padding: 18px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Authentication is initialised before dashboard data is loaded or shown.
initialise_auth_state()

# No data, dashboard page, or export is available until Firebase confirms
# both a valid session and a verified email address.
if not st.session_state.get("authenticated"):
    show_auth_page()
    st.stop()

current_user = require_verified_user()

if current_user is None:
    show_auth_page()
    st.stop()

# =========================
# AUTHENTICATED SIDEBAR
# =========================
st.sidebar.markdown("---")
st.sidebar.caption("Signed in as")
st.sidebar.write(current_user.get("email", "Verified user"))

if st.sidebar.button("Log out", use_container_width=True):
    logout_user()
    st.rerun()

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"],
)

df = load_data(uploaded_file)

# =========================
# SIDEBAR EXPORT SECTION
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("Export / Download")

st.sidebar.download_button(
    label="Download Uploaded Dataset",
    data=df.to_csv(index=False),
    file_name="customer_financial_behaviour_dataset.csv",
    mime="text/csv",
)

dataset_summary_text = f"""
Customer Financial Behaviour Dataset Summary

Total Records: {df.shape[0]:,}
Total Attributes: {df.shape[1]}
Missing Cells: {df.isna().sum().sum():,}
Numeric Columns: {len(df.select_dtypes(include=['int64', 'float64']).columns)}
Categorical Columns: {len(df.select_dtypes(include=['object', 'category', 'bool']).columns)}

Column Names:
{', '.join(df.columns)}
"""

st.sidebar.download_button(
    label="Download Dataset Summary",
    data=dataset_summary_text,
    file_name="dataset_summary.txt",
    mime="text/plain",
)

with st.sidebar.expander("Copy Dataset Summary"):
    st.text_area(
        "Copy this summary",
        dataset_summary_text,
        height=250,
    )

# =========================
# HEADER
# =========================
st.markdown(
    """
    <div class="glass-card">
        <h1>Customer Financial Behaviour Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Overview",
        "EDA",
        "K-Means Clustering",
        "Prediction Model",
        "Conclusion",
    ],
)

if page == "Home":
    show_home(df)

elif page == "Dataset Overview":
    show_dataset_overview(df)

elif page == "EDA":
    show_eda(df, theme)

elif page == "K-Means Clustering":
    show_pyspark_kmeans(df, theme)

elif page == "Prediction Model":
    show_random_forest(df, theme)

elif page == "Conclusion":
    show_conclusion(df)