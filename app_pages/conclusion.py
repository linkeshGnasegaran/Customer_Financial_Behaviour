import streamlit as st
import pandas as pd


def describe_strength(value):
    value = abs(value)

    if value >= 0.80:
        return "very strong"
    elif value >= 0.60:
        return "strong"
    elif value >= 0.40:
        return "moderate"
    elif value >= 0.20:
        return "weak"
    else:
        return "very weak"


def describe_model(score):
    if score >= 0.90:
        return "excellent"
    elif score >= 0.80:
        return "strong"
    elif score >= 0.70:
        return "moderate"
    elif score >= 0.60:
        return "acceptable"
    else:
        return "limited"


def add_line(lines, text):
    lines.append(text)
    st.write(text)


def show_conclusion(df):
    st.header("Dynamic Data-Driven Conclusion")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    rows = df.shape[0]
    cols = df.shape[1]
    missing_cells = int(df.isna().sum().sum())

    report_lines = []
    report_lines.append("DYNAMIC DATA-DRIVEN CONCLUSION REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")

    st.subheader("1. Dataset Structure Findings")

    add_line(
        report_lines,
        f"The uploaded dataset contains {rows:,} records and {cols} attributes."
    )

    add_line(
        report_lines,
        f"The system detected {len(numeric_cols)} numeric attributes and {len(categorical_cols)} categorical attributes."
    )

    if missing_cells == 0:
        add_line(
            report_lines,
            "No missing values were detected in the dataset."
        )
    else:
        missing_percent = (missing_cells / (rows * cols)) * 100
        add_line(
            report_lines,
            f"The dataset contains {missing_cells:,} missing values, representing {missing_percent:.2f}% of all cells."
        )

    report_lines.append("")

    st.subheader("2. Numeric Data Findings")

    if len(numeric_cols) > 0:
        numeric_summary = df[numeric_cols].describe().round(3)
        st.dataframe(numeric_summary, use_container_width=True)

        st.download_button(
            "Download Numeric Summary",
            numeric_summary.to_csv(),
            file_name="dynamic_numeric_summary.csv",
            mime="text/csv"
        )

        means = numeric_summary.loc["mean"].sort_values(ascending=False)
        stds = numeric_summary.loc["std"].sort_values(ascending=False)

        highest_mean_col = means.index[0]
        lowest_mean_col = means.index[-1]
        highest_std_col = stds.index[0]

        add_line(
            report_lines,
            f"The numeric attribute with the highest average value is {highest_mean_col} with a mean of {means.iloc[0]:,.3f}."
        )

        add_line(
            report_lines,
            f"The numeric attribute with the lowest average value is {lowest_mean_col} with a mean of {means.iloc[-1]:,.3f}."
        )

        add_line(
            report_lines,
            f"The attribute with the highest variation is {highest_std_col}, with a standard deviation of {stds.iloc[0]:,.3f}."
        )

    else:
        add_line(
            report_lines,
            "No numeric attributes were detected, so numeric summary analysis could not be performed."
        )

    report_lines.append("")

    st.subheader("3. Categorical Data Findings")

    if len(categorical_cols) > 0:
        categorical_summary_rows = []

        for col in categorical_cols:
            top_value = df[col].value_counts().idxmax()
            top_count = int(df[col].value_counts().max())
            unique_count = int(df[col].nunique())

            categorical_summary_rows.append({
                "Column": col,
                "Most Frequent Value": top_value,
                "Frequency": top_count,
                "Unique Values": unique_count
            })

        categorical_summary = pd.DataFrame(categorical_summary_rows)
        st.dataframe(categorical_summary, use_container_width=True)

        st.download_button(
            "Download Categorical Summary",
            categorical_summary.to_csv(index=False),
            file_name="dynamic_categorical_summary.csv",
            mime="text/csv"
        )

        for _, row in categorical_summary.head(5).iterrows():
            add_line(
                report_lines,
                f"For {row['Column']}, the most frequent value is {row['Most Frequent Value']} with {row['Frequency']:,} records."
            )

    else:
        add_line(
            report_lines,
            "No categorical attributes were detected, so categorical pattern analysis could not be performed."
        )

    report_lines.append("")

    st.subheader("4. Correlation Findings")

    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()

        corr_pairs = corr_matrix.unstack().reset_index()
        corr_pairs.columns = ["Variable 1", "Variable 2", "Correlation"]
        corr_pairs = corr_pairs[corr_pairs["Variable 1"] != corr_pairs["Variable 2"]]
        corr_pairs["Absolute Correlation"] = corr_pairs["Correlation"].abs()
        corr_pairs = corr_pairs.sort_values(by="Absolute Correlation", ascending=False)
        corr_pairs = corr_pairs.drop_duplicates(subset=["Absolute Correlation"])
        corr_pairs = corr_pairs.round(4)

        top_corr = corr_pairs.head(10)
        st.dataframe(top_corr, use_container_width=True)

        st.download_button(
            "Download Correlation Findings",
            top_corr.to_csv(index=False),
            file_name="dynamic_correlation_findings.csv",
            mime="text/csv"
        )

        strongest = top_corr.iloc[0]
        corr_strength = describe_strength(strongest["Correlation"])

        add_line(
            report_lines,
            f"The strongest relationship identified is between {strongest['Variable 1']} and {strongest['Variable 2']}."
        )

        add_line(
            report_lines,
            f"The correlation value is {strongest['Correlation']:.3f}, indicating a {corr_strength} relationship."
        )

        if strongest["Correlation"] > 0:
            add_line(
                report_lines,
                f"As {strongest['Variable 1']} increases, {strongest['Variable 2']} tends to increase as well."
            )
        else:
            add_line(
                report_lines,
                f"As {strongest['Variable 1']} increases, {strongest['Variable 2']} tends to decrease."
            )

    else:
        add_line(
            report_lines,
            "Correlation analysis could not be performed because fewer than two numeric attributes were detected."
        )

    report_lines.append("")

    st.subheader("5. Clustering Findings")

    if "cluster_profile" in st.session_state:
        cluster_profile = st.session_state["cluster_profile"]
        cluster_counts = st.session_state.get("cluster_counts")
        k_value = st.session_state.get("k_value")
        cluster_features = st.session_state.get("cluster_features", [])

        add_line(
            report_lines,
            f"K-Means clustering separated the dataset into {k_value} groups."
        )

        if cluster_counts is not None:
            st.dataframe(cluster_counts, use_container_width=True)

            st.download_button(
                "Download Cluster Counts",
                cluster_counts.to_csv(index=False),
                file_name="dynamic_cluster_counts.csv",
                mime="text/csv"
            )

            largest_row = cluster_counts.loc[cluster_counts["Count"].idxmax()]
            smallest_row = cluster_counts.loc[cluster_counts["Count"].idxmin()]

            add_line(
                report_lines,
                f"The largest group is Cluster {largest_row['Cluster']} with {int(largest_row['Count']):,} records."
            )

            add_line(
                report_lines,
                f"The smallest group is Cluster {smallest_row['Cluster']} with {int(smallest_row['Count']):,} records."
            )

        st.dataframe(cluster_profile, use_container_width=True)

        st.download_button(
            "Download Cluster Profile",
            cluster_profile.to_csv(),
            file_name="dynamic_cluster_profile.csv",
            mime="text/csv"
        )

        for feature in cluster_features[:5]:
            if feature in cluster_profile.columns:
                highest_cluster = cluster_profile[feature].idxmax()
                lowest_cluster = cluster_profile[feature].idxmin()

                highest_value = cluster_profile.loc[highest_cluster, feature]
                lowest_value = cluster_profile.loc[lowest_cluster, feature]

                add_line(
                    report_lines,
                    f"For {feature}, Cluster {highest_cluster} recorded the highest average value ({highest_value:,.3f}), while Cluster {lowest_cluster} recorded the lowest average value ({lowest_value:,.3f})."
                )

    else:
        add_line(
            report_lines,
            "Clustering findings are not available yet because the K-Means page has not been executed in this session."
        )

    report_lines.append("")

    st.subheader("6. Prediction Model Findings")

    if "rf_accuracy" in st.session_state:
        accuracy = st.session_state["rf_accuracy"]
        precision = st.session_state["rf_precision"]
        recall = st.session_state["rf_recall"]
        f1 = st.session_state["rf_f1"]
        target = st.session_state.get("rf_target", "selected target")
        features = st.session_state.get("rf_features", [])

        metrics_table = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Value": [accuracy, precision, recall, f1]
        })

        st.dataframe(metrics_table, use_container_width=True)

        st.download_button(
            "Download Prediction Metrics",
            metrics_table.to_csv(index=False),
            file_name="dynamic_prediction_metrics.csv",
            mime="text/csv"
        )

        add_line(
            report_lines,
            f"The prediction model was trained using {len(features)} selected attributes to predict {target}."
        )

        add_line(
            report_lines,
            f"The model achieved an accuracy of {accuracy:.3f}, which indicates {describe_model(accuracy)} predictive performance."
        )

        add_line(
            report_lines,
            f"The precision value is {precision:.3f}, the recall value is {recall:.3f}, and the F1 score is {f1:.3f}."
        )

        if f1 >= 0.70:
            add_line(
                report_lines,
                "The F1 score suggests that the model maintains a useful balance between precision and recall."
            )
        else:
            add_line(
                report_lines,
                "The F1 score suggests that further feature engineering or model tuning may improve prediction performance."
            )

    else:
        add_line(
            report_lines,
            "Prediction findings are not available yet because the Prediction Model page has not been executed in this session."
        )

    report_lines.append("")

    st.subheader("7. Final Generated Conclusion")

    generated_conclusion = []

    generated_conclusion.append(
        f"The dataset contains {rows:,} records across {cols} attributes, providing a structured basis for analysis."
    )

    if missing_cells == 0:
        generated_conclusion.append(
            "The absence of missing values improves the reliability of the analysis."
        )
    else:
        generated_conclusion.append(
            f"The presence of {missing_cells:,} missing values indicates that preprocessing is required before final modelling."
        )

    if len(numeric_cols) >= 2:
        generated_conclusion.append(
            f"The strongest detected numeric relationship is between {strongest['Variable 1']} and {strongest['Variable 2']} with a correlation of {strongest['Correlation']:.3f}."
        )

    if "cluster_profile" in st.session_state:
        generated_conclusion.append(
            f"The clustering process identified {k_value} distinguishable groups in the dataset."
        )

    if "rf_accuracy" in st.session_state:
        generated_conclusion.append(
            f"The prediction model achieved {describe_model(accuracy)} performance with an accuracy of {accuracy:.3f}."
        )

    final_text = " ".join(generated_conclusion)

    st.write(final_text)

    report_lines.append("7. Final Generated Conclusion")
    report_lines.append(final_text)
    report_lines.append("")

    st.subheader("8. Generated Recommendations")

    recommendations = []

    if missing_cells > 0:
        recommendations.append(
            f"Clean or impute the {missing_cells:,} missing values before relying on final analytical results."
        )

    if len(numeric_cols) >= 2:
        recommendations.append(
            f"Further investigate the relationship between {strongest['Variable 1']} and {strongest['Variable 2']} because it produced the strongest correlation."
        )

    if "cluster_profile" in st.session_state and cluster_features:
        recommendations.append(
            f"Use the cluster differences in {cluster_features[0]} to interpret group-level patterns in the dataset."
        )

    if "rf_accuracy" in st.session_state:
        recommendations.append(
            f"Improve the prediction model for {target} by testing additional features, parameter tuning, or alternative algorithms."
        )

    if not recommendations:
        recommendations.append(
            "Run EDA, clustering, and prediction modules to generate stronger recommendations."
        )

    for i, rec in enumerate(recommendations, start=1):
        st.write(f"{i}. {rec}")
        report_lines.append(f"{i}. {rec}")

    final_report_text = "\n".join(report_lines)

    st.subheader("9. Export / Copy Generated Report")

    st.text_area(
        "Copy Generated Conclusion Report",
        final_report_text,
        height=450
    )

    st.download_button(
        "Download Generated Conclusion Report",
        final_report_text,
        file_name="dynamic_data_driven_conclusion_report.txt",
        mime="text/plain"
    )