import streamlit as st

from utils.data_loader import load_data
from utils.theme import load_theme

from app_pages.home import show_home
from app_pages.dataset_overview import show_dataset_overview
from app_pages.eda import show_eda
from app_pages.pyspark_kmeans import show_pyspark_kmeans
from app_pages.random_forest import show_random_forest
from app_pages.conclusion import show_conclusion

st.set_page_config(
    page_title="Customer Financial Behaviour",
    layout="wide"
)

theme = st.sidebar.selectbox(
    "Theme Mode",
    ["System Mode", "Light Mode", "Dark Mode"]
)

load_theme(theme)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
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
    mime="text/csv"
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
    mime="text/plain"
)

with st.sidebar.expander("Copy Dataset Summary"):
    st.text_area(
        "Copy this summary",
        dataset_summary_text,
        height=250
    )

# =========================
# HEADER
# =========================
st.markdown("""
<div class="glass-card">
    <h1>Customer Financial Behaviour Dashboard</h1>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Overview",
        "EDA",
        "K-Means Clustering",
        "Prediction Model",
        "Conclusion"
    ]
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