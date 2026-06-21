import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from utils.charts import style_chart, get_chart_colors


def show_random_forest(df, theme):
    st.header("Automatic Prediction Model")

    colors = get_chart_colors(theme)

    st.markdown("""
    <div class="glass-card">
        <p>
        This prediction module lets the user select a target variable from the uploaded dataset.
        The system automatically detects numeric and categorical features, then trains a
        Random Forest classification model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    all_columns = df.columns.tolist()

    target = st.selectbox(
        "Select target variable for prediction",
        all_columns
    )

    model_data = df.dropna().copy()

    if target not in model_data.columns:
        st.error("Selected target column not found.")
        st.stop()

    y = model_data[target]

    if y.nunique() < 2:
        st.error("Target variable must have at least two classes.")
        st.stop()

    if y.nunique() > 20:
        st.warning(
            "This target has many unique values. Random Forest classification works best with categorical or binary target variables."
        )
        st.stop()

    X = model_data.drop(columns=[target])

    numeric_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    selected_numeric = st.multiselect(
        "Select numeric features",
        numeric_features,
        default=numeric_features[:min(6, len(numeric_features))]
    )

    selected_categorical = st.multiselect(
        "Select categorical features",
        categorical_features,
        default=categorical_features[:min(3, len(categorical_features))]
    )

    selected_features = selected_numeric + selected_categorical

    if len(selected_features) == 0:
        st.warning("Please select at least one feature.")
        st.stop()

    max_rows = st.slider(
        "Rows used for prediction model",
        min_value=1000,
        max_value=min(135000, len(model_data)),
        value=min(50000, len(model_data)),
        step=1000
    )

    sampled_data = model_data[selected_features + [target]].sample(
        min(max_rows, len(model_data)),
        random_state=42
    )

    X = sampled_data[selected_features]
    y = sampled_data[target]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), selected_numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), selected_categorical)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

    with st.spinner("Training Random Forest model..."):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    average_type = "binary" if y.nunique() == 2 else "weighted"

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        average=average_type,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        average=average_type,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        average=average_type,
        zero_division=0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{accuracy:.3f}")
    c2.metric("Precision", f"{precision:.3f}")
    c3.metric("Recall", f"{recall:.3f}")
    c4.metric("F1 Score", f"{f1:.3f}")

    # ===============================
    # MODEL METRICS EXPORT
    # ===============================
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy, precision, recall, f1]
    })

    st.subheader("Model Metrics Table")
    st.dataframe(metrics_df, use_container_width=True)

    st.download_button(
        "Download Model Metrics CSV",
        metrics_df.to_csv(index=False),
        file_name="random_forest_model_metrics.csv",
        mime="text/csv"
    )

    # ===============================
    # CONFUSION MATRIX
    # ===============================
    cm_array = confusion_matrix(y_test, predictions)

    st.subheader("Confusion Matrix")

    labels = sorted(y.unique().tolist())

    cm = pd.DataFrame(
        cm_array,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels]
    )

    st.dataframe(cm, use_container_width=True)

    st.download_button(
        "Download Confusion Matrix CSV",
        cm.to_csv(),
        file_name="random_forest_confusion_matrix.csv",
        mime="text/csv"
    )

    st.subheader("Confusion Matrix Chart")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(cm_array)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([str(label) for label in labels], rotation=30)
    ax.set_yticklabels([str(label) for label in labels])

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                cm_array[i, j],
                ha="center",
                va="center",
                color=colors["text"],
                fontsize=11,
                fontweight="bold"
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    style_chart(fig, ax, "Confusion Matrix", theme)
    st.pyplot(fig)

    # ===============================
    # CLASSIFICATION REPORT
    # ===============================
    st.subheader("Classification Report")

    report_text = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    st.text(report_text)

    report_dict = classification_report(
        y_test,
        predictions,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report_dict).transpose()

    st.download_button(
        "Download Classification Report CSV",
        report_df.to_csv(),
        file_name="random_forest_classification_report.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Classification Report TXT",
        report_text,
        file_name="random_forest_classification_report.txt",
        mime="text/plain"
    )

    # ===============================
    # PREDICTION RESULTS
    # ===============================
    prediction_results = X_test.copy()
    prediction_results["Actual"] = y_test.values
    prediction_results["Predicted"] = predictions

    st.subheader("Prediction Results Preview")
    st.dataframe(
        prediction_results.head(50),
        use_container_width=True
    )

    st.download_button(
        "Download Prediction Results CSV",
        prediction_results.to_csv(index=False),
        file_name="random_forest_prediction_results.csv",
        mime="text/csv"
    )

    # ===============================
    # COPY / EXPORT SUMMARY
    # ===============================
    selected_feature_text = ", ".join(selected_features)

    rf_summary = f"""
Random Forest Prediction Summary

Target Variable:
{target}

Selected Features:
{selected_feature_text}

Rows Used:
{len(sampled_data):,}

Model Performance:
Accuracy: {accuracy:.3f}
Precision: {precision:.3f}
Recall: {recall:.3f}
F1 Score: {f1:.3f}

Interpretation:
The Random Forest model was trained to predict {target}.
The model achieved an accuracy of {accuracy:.3f}, precision of {precision:.3f},
recall of {recall:.3f}, and F1 score of {f1:.3f}.
These results indicate how effectively the selected features support prediction
of the selected target variable.
"""

    st.subheader("Random Forest Findings Summary")

    st.text_area(
        "Copy Random Forest Findings Summary",
        rf_summary,
        height=300
    )

    st.download_button(
        "Download Random Forest Findings Summary",
        rf_summary,
        file_name="random_forest_findings_summary.txt",
        mime="text/plain"
    )

    # ===============================
    # SESSION STATE FOR CONCLUSION
    # ===============================
    st.session_state["rf_accuracy"] = accuracy
    st.session_state["rf_precision"] = precision
    st.session_state["rf_recall"] = recall
    st.session_state["rf_f1"] = f1
    st.session_state["rf_target"] = target
    st.session_state["rf_features"] = selected_features
    st.session_state["rf_summary"] = rf_summary
    st.session_state["rf_metrics_df"] = metrics_df
    st.session_state["rf_confusion_matrix"] = cm