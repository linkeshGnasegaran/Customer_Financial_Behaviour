import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

from utils.charts import style_chart, get_chart_colors


def fig_to_png(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    return buffer


def show_eda(df, theme):
    st.header("Exploratory Data Analysis")

    colors = get_chart_colors(theme)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    st.subheader("Detected Columns")
    detected_info = f"""
Numeric Columns: {len(numeric_cols)}
Categorical Columns: {len(categorical_cols)}

Numeric Column Names:
{", ".join(numeric_cols)}

Categorical Column Names:
{", ".join(categorical_cols)}
"""

    st.text_area("Copy Detected Column Summary", detected_info, height=180)

    st.download_button(
        "Download Detected Column Summary",
        detected_info,
        file_name="eda_detected_columns.txt",
        mime="text/plain"
    )

    st.write(f"Numeric columns: {len(numeric_cols)}")
    st.write(f"Categorical columns: {len(categorical_cols)}")

    st.subheader("Numeric Column Analysis")

    if len(numeric_cols) == 0:
        st.warning("No numeric columns found.")
    else:
        numeric_summary = df[numeric_cols].describe().round(3)

        st.write("Numeric Summary Statistics")
        st.dataframe(numeric_summary, use_container_width=True)

        st.download_button(
            "Download Numeric Summary CSV",
            numeric_summary.to_csv(),
            file_name="numeric_summary_statistics.csv",
            mime="text/csv"
        )

        selected_numeric = st.multiselect(
            "Select numeric columns for histogram",
            numeric_cols,
            default=numeric_cols[:min(5, len(numeric_cols))]
        )

        for column in selected_numeric:
            st.subheader(f"{column} Distribution")

            hist_data = df[column].dropna().reset_index(drop=True).to_frame()
            hist_data.columns = [column]

            st.download_button(
                f"Download {column} Histogram Data",
                hist_data.to_csv(index=False),
                file_name=f"{column}_histogram_data.csv",
                mime="text/csv"
            )

            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.hist(df[column].dropna(), bins=40, color=colors["bar"])
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            style_chart(fig, ax, f"{column} Distribution", theme)
            st.pyplot(fig)

            st.download_button(
                f"Save {column} Chart as PNG",
                fig_to_png(fig),
                file_name=f"{column}_distribution.png",
                mime="image/png"
            )

    st.subheader("Categorical Column Analysis")

    if len(categorical_cols) == 0:
        st.warning("No categorical columns found.")
    else:
        selected_categorical = st.multiselect(
            "Select categorical columns for bar chart",
            categorical_cols,
            default=categorical_cols[:min(5, len(categorical_cols))]
        )

        for column in selected_categorical:
            st.subheader(f"{column} Bar Chart")

            counts = df[column].value_counts().head(15).reset_index()
            counts.columns = [column, "Count"]

            st.dataframe(counts, use_container_width=True)

            st.download_button(
                f"Download {column} Frequency Table",
                counts.to_csv(index=False),
                file_name=f"{column}_frequency_table.csv",
                mime="text/csv"
            )

            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar(counts[column].astype(str), counts["Count"], color=colors["bar"])
            ax.set_xlabel(column)
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=30)
            style_chart(fig, ax, f"{column} Bar Chart", theme)
            st.pyplot(fig)

            st.download_button(
                f"Save {column} Bar Chart as PNG",
                fig_to_png(fig),
                file_name=f"{column}_bar_chart.png",
                mime="image/png"
            )

    st.subheader("Scatter Plot Analysis")

    if len(numeric_cols) >= 2:
        x_col = st.selectbox("Select X-axis", numeric_cols, index=0)
        y_col = st.selectbox("Select Y-axis", numeric_cols, index=1)

        temp = df[[x_col, y_col]].dropna()

        if len(temp) > 0:
            sample_df = temp.sample(min(5000, len(temp)), random_state=42)

            st.download_button(
                f"Download Scatter Plot Data: {x_col} vs {y_col}",
                sample_df.to_csv(index=False),
                file_name=f"{x_col}_vs_{y_col}_scatter_data.csv",
                mime="text/csv"
            )

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(
                sample_df[x_col],
                sample_df[y_col],
                s=8,
                alpha=0.5,
                color=colors["scatter"]
            )

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            style_chart(fig, ax, f"{x_col} vs {y_col}", theme)
            st.pyplot(fig)

            st.download_button(
                f"Save Scatter Plot {x_col} vs {y_col} as PNG",
                fig_to_png(fig),
                file_name=f"{x_col}_vs_{y_col}_scatter_plot.png",
                mime="image/png"
            )

    st.subheader("Correlation Matrix")

    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().round(3)

        st.dataframe(corr_matrix, use_container_width=True)

        st.download_button(
            "Download Correlation Matrix CSV",
            corr_matrix.to_csv(),
            file_name="correlation_matrix.csv",
            mime="text/csv"
        )

        corr_pairs = (
            df[numeric_cols]
            .corr()
            .unstack()
            .reset_index()
        )

        corr_pairs.columns = ["Variable 1", "Variable 2", "Correlation"]
        corr_pairs = corr_pairs[corr_pairs["Variable 1"] != corr_pairs["Variable 2"]]
        corr_pairs["Absolute Correlation"] = corr_pairs["Correlation"].abs()
        corr_pairs = corr_pairs.sort_values(by="Absolute Correlation", ascending=False)
        corr_pairs = corr_pairs.drop_duplicates(subset=["Absolute Correlation"])
        corr_pairs = corr_pairs.round(4)

        top_corr = corr_pairs.head(10)

        st.subheader("Top Correlation Relationships")
        st.dataframe(top_corr, use_container_width=True)

        st.download_button(
            "Download Top Correlation Relationships CSV",
            top_corr.to_csv(index=False),
            file_name="top_correlation_relationships.csv",
            mime="text/csv"
        )

        strongest = corr_pairs.iloc[0]

        eda_findings = f"""
EDA Findings Summary

Dataset contains {df.shape[0]:,} records and {df.shape[1]} attributes.
The system detected {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns.

Strongest Correlation:
Variable 1: {strongest["Variable 1"]}
Variable 2: {strongest["Variable 2"]}
Correlation: {strongest["Correlation"]:.4f}

Interpretation:
The strongest relationship identified in this dataset is between {strongest["Variable 1"]} and {strongest["Variable 2"]}.
A correlation value closer to +1 indicates a strong positive relationship, while a value closer to -1 indicates a strong negative relationship.
"""

        st.text_area(
            "Copy EDA Findings Summary",
            eda_findings,
            height=250
        )

        st.download_button(
            "Download EDA Findings Summary",
            eda_findings,
            file_name="eda_findings_summary.txt",
            mime="text/plain"
        )

        st.session_state["eda_corr_pairs"] = corr_pairs
        st.session_state["numeric_cols"] = numeric_cols
        st.session_state["categorical_cols"] = categorical_cols
        st.session_state["eda_findings"] = eda_findings