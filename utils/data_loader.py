import streamlit as st
import pandas as pd


@st.cache_data
def read_csv_file(file):
    return pd.read_csv(file)


def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = read_csv_file(uploaded_file)
        else:
            df = read_csv_file("data/origination_data-2.csv")

        df.columns = df.columns.str.strip()
        return df

    except FileNotFoundError:
        st.error("Dataset not found. Please upload a CSV dataset.")
        st.stop()