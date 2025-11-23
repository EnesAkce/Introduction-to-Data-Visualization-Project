import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# -----------------------------------------------------------
# DATA LOAD
# -----------------------------------------------------------
@st.cache_data
def load_data():
    # CSV file is in the same folder
    df = pd.read_csv("turkey_earthquakes(1915-2024_feb).csv")
    df["Olus tarihi"] = pd.to_datetime(df["Olus tarihi"], errors="coerce")
    return df

df = load_data()

st.title("Member 3 - Relationship Analysis & Clustering")

# -----------------------------------------------------------
# CHART 7 – PARALLEL COORDINATES
# -----------------------------------------------------------
st.subheader("1. Relationship Between Magnitude Types – Parallel Coordinates")

df_mag = df[["MD", "ML", "Mw", "xM"]].dropna()

fig7 = px.parallel_coordinates(
    df_mag,
    color="xM",
    color_continuous_scale=px.colors.sequential.Viridis,
    title="Relationship Between MD – ML – Mw Magnitude Types"
)

st.plotly_chart(fig7, use_container_width=True)

# -----------------------------------------------------------
# CHART 8 – KMEANS CLUSTERING
# -----------------------------------------------------------
st.subheader("2. Earthquake K-Means Clustering")

k = st.slider("Number of Clusters (k):", 2, 6, 3)

df_cluster = df[["Enlem", "Boylam", "Derinlik", "xM"]].dropna()

kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(df_cluster)

df_cluster["cluster"] = labels

fig8 = px.scatter_3d(
    df_cluster,
    x="Boylam",
    y="Enlem",
    z="Derinlik",
    color="cluster",
    size="xM",
    title=f"K={k} K-means Earthquake Clusters"
)

st.plotly_chart(fig8, use_container_width=True)

# -----------------------------------------------------------
# CHART 9 – BUBBLE CHART
# -----------------------------------------------------------
st.subheader("3. Magnitude vs Depth (Bubble Chart)")

year_range = st.slider(
    "Select Year Range:",
    int(df["Olus tarihi"].dt.year.min()),
    int(df["Olus tarihi"].dt.year.max()),
    (1990, 2024)
)

df_bubble = df[
    (df["Olus tarihi"].dt.year >= year_range[0]) &
    (df["Olus tarihi"].dt.year <= year_range[1]) &
    (df["Derinlik"].notna())
]

fig9 = px.scatter(
    df_bubble,
    x="Derinlik",
    y="xM",
    size="xM",
    color="xM",
    hover_data=["Yer", "Olus tarihi"],
    title=f"Magnitude vs Depth from {year_range[0]} to {year_range[1]}"
)

st.plotly_chart(fig9, use_container_width=True)

st.info("End of Member 3 section.")
