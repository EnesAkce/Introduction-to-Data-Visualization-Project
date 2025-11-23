import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans

# -----------------------------------------------------------
# DATA LOAD
# -----------------------------------------------------------
@st.cache_data
def load_data():
    # CSV dosyası aynı klasörde
    df = pd.read_csv("turkey_earthquakes(1915-2024_feb).csv")
    df["Olus tarihi"] = pd.to_datetime(df["Olus tarihi"], errors="coerce")
    return df

df = load_data()

st.title("Üye 3 - İlişki Analizi ve Kümelendirme")

# -----------------------------------------------------------
# GRAFIK 7 – PARALLEL COORDINATES
# -----------------------------------------------------------
st.subheader("1. Magnitüd Türleri Arası İlişki – Parallel Coordinates")

df_mag = df[["MD", "ML", "Mw", "xM"]].dropna()

fig7 = px.parallel_coordinates(
    df_mag,
    color="xM",
    color_continuous_scale=px.colors.sequential.Viridis,
    title="MD – ML – Mw Büyüklük Türleri Arasındaki İlişki"
)

st.plotly_chart(fig7, use_container_width=True)

# -----------------------------------------------------------
# GRAFIK 8 – KMEANS CLUSTERING
# -----------------------------------------------------------
st.subheader("2. Depremlerin K-Means Kümelenmesi")

k = st.slider("Küme Sayısı (k):", 2, 6, 3)

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
    title=f"K={k} K-means Deprem Kümeleri"
)

st.plotly_chart(fig8, use_container_width=True)

# -----------------------------------------------------------
# GRAFIK 9 – BUBBLE CHART
# -----------------------------------------------------------
st.subheader("3. Magnitüd – Derinlik İlişkisi (Bubble Chart)")

year_range = st.slider(
    "Yıl Aralığı Seç:",
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
    title=f"{year_range[0]}–{year_range[1]} Arasında Derinlik – Büyüklük İlişkisi"
)

st.plotly_chart(fig9, use_container_width=True)

st.info("Üye 3 bölümünün sonu.")