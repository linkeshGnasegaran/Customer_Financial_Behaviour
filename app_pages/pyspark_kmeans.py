import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from utils.charts import style_chart, get_chart_colors


def fig_to_png(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    buffer.seek(0)
    return buffer


def show_pyspark_kmeans(df, theme):
    st.header("Automatic K-Means Clustering")

    colors = get_chart_colors(theme)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("At least two numeric columns are required for clustering.")
        st.stop()

    st.markdown("""
    <div class="glass-card">
        <p>
        This clustering module automatically detects numeric columns from the uploaded dataset.
        The user can select any numeric features for K-Means clustering.
        </p>
    </div>
    """, unsafe_allow_html=True)

    selected_features = st.multiselect(
        "Select features for clustering",
        numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))]
    )

    if len(selected_features) < 2:
        st.warning("Please select at least two numeric features.")
        st.stop()

    k = st.slider("Number of Clusters", min_value=2, max_value=8, value=3)

    model_df = df[selected_features].dropna().copy()

    max_rows = st.slider(
        "Rows used for clustering",
        min_value=1000,
        max_value=min(135000, len(model_df)),
        value=min(50000, len(model_df)),
        step=1000
    )

    model_df = model_df.sample(min(max_rows, len(model_df)), random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_df)

    st.subheader("Elbow Method: SSE vs K")

    sse_values = []
    k_values = list(range(2, 9))

    for temp_k in k_values:
        temp_model = KMeans(n_clusters=temp_k, random_state=42, n_init=10)
        temp_model.fit(X_scaled)
        sse_values.append(temp_model.inertia_)

    elbow_df = pd.DataFrame({"K": k_values, "SSE": sse_values})

    st.dataframe(elbow_df, use_container_width=True)

    st.download_button(
        "Download Elbow Method Data",
        elbow_df.to_csv(index=False),
        file_name="kmeans_elbow_method.csv",
        mime="text/csv"
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(elbow_df["K"], elbow_df["SSE"], marker="o", color=colors["bar"])
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("SSE")
    style_chart(fig, ax, "Elbow Method", theme)
    st.pyplot(fig)

    st.download_button(
        "Save Elbow Method Chart as PNG",
        fig_to_png(fig),
        file_name="kmeans_elbow_method.png",
        mime="image/png"
    )

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    model_df["cluster"] = kmeans.fit_predict(X_scaled)

    st.subheader("Cluster Counts")

    cluster_counts = model_df["cluster"].value_counts().sort_index()
    cluster_counts_df = pd.DataFrame({
        "Cluster": cluster_counts.index,
        "Count": cluster_counts.values
    })

    st.dataframe(cluster_counts_df, use_container_width=True)

    st.download_button(
        "Download Cluster Counts",
        cluster_counts_df.to_csv(index=False),
        file_name="kmeans_cluster_counts.csv",
        mime="text/csv"
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        cluster_counts_df["Cluster"].astype(str),
        cluster_counts_df["Count"],
        color=colors["bar"]
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    style_chart(fig, ax, "Cluster Counts", theme)
    st.pyplot(fig)

    st.download_button(
        "Save Cluster Counts Chart as PNG",
        fig_to_png(fig),
        file_name="kmeans_cluster_counts.png",
        mime="image/png"
    )

    st.subheader("Cluster Profile / Mean Values")

    cluster_profile = model_df.groupby("cluster")[selected_features].mean().round(2)
    st.dataframe(cluster_profile, use_container_width=True)

    st.download_button(
        "Download Cluster Profile",
        cluster_profile.to_csv(),
        file_name="kmeans_cluster_profile.csv",
        mime="text/csv"
    )

    st.subheader("Clustered Dataset Preview")

    clustered_preview = model_df.head(50)
    st.dataframe(clustered_preview, use_container_width=True)

    st.download_button(
        "Download Full Clustered Dataset",
        model_df.to_csv(index=False),
        file_name="kmeans_clustered_dataset.csv",
        mime="text/csv"
    )

    st.subheader("Cluster Scatter Plot with Centroids")

    x_axis = st.selectbox("Select X-axis for cluster plot", selected_features, index=0)
    y_axis = st.selectbox("Select Y-axis for cluster plot", selected_features, index=1)

    centroid_df = model_df.groupby("cluster")[[x_axis, y_axis]].mean().reset_index()
    plot_df = model_df.sample(min(8000, len(model_df)), random_state=42)

    st.download_button(
        "Download Scatter Plot Data",
        plot_df[[x_axis, y_axis, "cluster"]].to_csv(index=False),
        file_name="kmeans_scatter_plot_data.csv",
        mime="text/csv"
    )

    st.download_button(
        "Download Centroid Values",
        centroid_df.to_csv(index=False),
        file_name="kmeans_centroid_values.csv",
        mime="text/csv"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        plot_df[x_axis],
        plot_df[y_axis],
        c=plot_df["cluster"],
        s=10,
        alpha=0.6
    )

    ax.scatter(
        centroid_df[x_axis],
        centroid_df[y_axis],
        marker="X",
        s=300,
        color=colors["centroid"],
        edgecolors="black",
        linewidths=1.5,
        label="Centroid"
    )

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.legend()

    style_chart(fig, ax, f"K-Means Clusters: {x_axis} vs {y_axis}", theme)
    st.pyplot(fig)

    st.download_button(
        "Save Cluster Scatter Plot as PNG",
        fig_to_png(fig),
        file_name=f"kmeans_{x_axis}_vs_{y_axis}_scatter.png",
        mime="image/png"
    )

    st.subheader("Centroid Values")

    centroid_values = model_df.groupby("cluster")[selected_features].mean().round(2)
    st.dataframe(centroid_values, use_container_width=True)

    highest_feature_notes = []

    for feature in selected_features:
        high_cluster = cluster_profile[feature].idxmax()
        low_cluster = cluster_profile[feature].idxmin()

        highest_feature_notes.append(
            f"For {feature}, Cluster {high_cluster} has the highest average value, "
            f"while Cluster {low_cluster} has the lowest average value."
        )

    kmeans_summary = f"""
K-Means Clustering Summary

Selected Features:
{", ".join(selected_features)}

Number of Clusters:
{k}

Rows Used:
{len(model_df):,}

Cluster Counts:
{cluster_counts_df.to_string(index=False)}

Cluster Interpretation:
{chr(10).join(highest_feature_notes)}

Scatter Plot Axes:
X-axis: {x_axis}
Y-axis: {y_axis}
"""

    st.subheader("K-Means Findings Summary")

    st.text_area(
        "Copy K-Means Findings Summary",
        kmeans_summary,
        height=300
    )

    st.download_button(
        "Download K-Means Findings Summary",
        kmeans_summary,
        file_name="kmeans_findings_summary.txt",
        mime="text/plain"
    )

    st.session_state["cluster_profile"] = cluster_profile
    st.session_state["cluster_counts"] = cluster_counts_df
    st.session_state["k_value"] = k
    st.session_state["cluster_features"] = selected_features
    st.session_state["cluster_x_axis"] = x_axis
    st.session_state["cluster_y_axis"] = y_axis
    st.session_state["kmeans_summary"] = kmeans_summary