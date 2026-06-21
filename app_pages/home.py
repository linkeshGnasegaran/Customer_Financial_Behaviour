import streamlit as st


def show_home(df):

    # Glass card CSS
    st.markdown("""
        <style>
        .glass-card {
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.25);
            border-radius: 16px;
            padding: 22px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        </style>
    """, unsafe_allow_html=True)

    # Project Overview Card
    st.markdown("""
        <div class="glass-card">
            <h2>Project Overview</h2>
        </div>
    """, unsafe_allow_html=True)

    st.write(
        "This system analyses customer financial behaviour using Big Data Analytics "
        "and Machine Learning. It includes Exploratory Data Analysis, PySpark "
        "K-Means customer segmentation, Random Forest default prediction, and "
        "an interactive Streamlit dashboard."
    )

    st.divider()

    # Metrics
    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", f"{df.shape[0]:,}")
    c2.metric("Total Attributes", df.shape[1])

    if "default_12m" in df.columns:
        default_rate = df["default_12m"].mean() * 100
        c3.metric("Default Rate", f"{default_rate:.2f}%")
    else:
        c3.metric("Default Rate", "N/A")

    st.divider()

    # Research Question
    st.subheader("Research Question")

    st.write(
        "How can machine learning techniques be used to analyse customer financial "
        "behaviour and identify meaningful customer segments for banking decision-making?"
    )

    # Hypothesis
    st.subheader("Hypothesis")

    st.write(
        "Customers with higher debt-to-income ratio, higher credit utilisation, "
        "and more credit inquiries are more likely to demonstrate risky financial behaviour."
    )