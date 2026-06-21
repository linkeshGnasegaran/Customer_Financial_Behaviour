import streamlit as st
import pandas as pd


def show_dataset_overview(df):

    st.header("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Cells", f"{df.isna().sum().sum():,}")

    # ===================================
    # DATASET INFORMATION
    # ===================================

    st.subheader("Dataset Information")

    dataset_info = f"""
Total Records: {df.shape[0]:,}
Total Columns: {df.shape[1]}
Missing Cells: {df.isna().sum().sum():,}

Columns:
{', '.join(df.columns)}
"""

    st.text_area(
        "Copy Dataset Information",
        dataset_info,
        height=180
    )

    st.download_button(
        "Download Dataset Information",
        dataset_info,
        file_name="dataset_information.txt",
        mime="text/plain"
    )

    # ===================================
    # SAMPLE RECORDS
    # ===================================

    st.subheader("Sample Records")

    sample_df = df.head(20)

    st.dataframe(
        sample_df,
        use_container_width=True
    )

    st.download_button(
        "Download Sample Records CSV",
        sample_df.to_csv(index=False),
        file_name="sample_records.csv",
        mime="text/csv"
    )

    # ===================================
    # DATA TYPES
    # ===================================

    st.subheader("Data Types")

    data_types = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values
    })

    st.dataframe(
        data_types,
        use_container_width=True
    )

    st.download_button(
        "Download Data Types CSV",
        data_types.to_csv(index=False),
        file_name="data_types.csv",
        mime="text/csv"
    )

    # ===================================
    # MISSING VALUES
    # ===================================

    st.subheader("Missing Values")

    missing = df.isna().sum().reset_index()
    missing.columns = [
        "Column",
        "Missing Values"
    ]

    st.dataframe(
        missing,
        use_container_width=True
    )

    st.download_button(
        "Download Missing Values CSV",
        missing.to_csv(index=False),
        file_name="missing_values.csv",
        mime="text/csv"
    )

    # ===================================
    # STATISTICAL SUMMARY
    # ===================================

    st.subheader("Statistical Summary")

    numeric_df = df.select_dtypes(
        include=["int64", "float64"]
    )

    if len(numeric_df.columns) > 0:

        stats_df = numeric_df.describe().round(2)

        st.dataframe(
            stats_df,
            use_container_width=True
        )

        st.download_button(
            "Download Statistical Summary CSV",
            stats_df.to_csv(),
            file_name="statistical_summary.csv",
            mime="text/csv"
        )

    else:
        st.warning(
            "No numeric columns available."
        )

    # ===================================
    # COLUMN LIST
    # ===================================

    st.subheader("Column Names")

    column_text = "\n".join(df.columns)

    st.text_area(
        "Copy Column Names",
        column_text,
        height=250
    )

    st.download_button(
        "Download Column Names",
        column_text,
        file_name="column_names.txt",
        mime="text/plain"
    )