import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Dashboard Analisis Nilai Siswa",
    layout="wide"
)

st.title("📊 Dashboard Analisis Nilai Siswa")

# ==================================================
# 1️⃣ UPLOAD DATA
# ==================================================
st.header("1️⃣ Upload Data")

uploaded_file = st.file_uploader(
    "Unggah file Excel (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Data Mentah")
    st.dataframe(df)

    # Identifikasi kolom numerik (soal)
    soal_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # Hitung total nilai
    df["Total_Nilai"] = df[soal_cols].sum(axis=1)

    # ==================================================
    # 2️⃣ KPI (Key Performance Indicator)
    # ==================================================
    st.header("2️⃣ Key Performance Indicator (KPI)")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Jumlah Siswa", len(df))
    col2.metric("Jumlah Soal", len(soal_cols))
    col3.metric("Rata-rata Total Nilai", round(df["Total_Nilai"].mean(), 2))
    col4.metric("Nilai Tertinggi", df["Total_Nilai"].max())

    # ==================================================
    # 3️⃣ DISTRIBUSI TOTAL NILAI
    # ==================================================
    st.header("3️⃣ Distribusi Total Nilai")

    fig, ax = plt.subplots()
    ax.hist(df["Total_Nilai"], bins=10)
    ax.set_xlabel("Total Nilai")
    ax.set_ylabel("Jumlah Siswa")
    ax.set_title("Distribusi Total Nilai Siswa")
    st.pyplot(fig)

    # ==================================================
    # 4️⃣ ANALISIS TINGKAT KESULITAN SOAL
    # ==================================================
    st.header("4️⃣ Analisis Tingkat Kesulitan Soal")

    tingkat_kesulitan = df[soal_cols].mean()

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    tingkat_kesulitan.plot(kind="bar", ax=ax2)
    ax2.set_ylabel("Rata-rata Skor")
    ax2.set_title("Tingkat Kesulitan Soal (Semakin Rendah = Semakin Sulit)")
    st.pyplot(fig2)

    # ==================================================
    # 5️⃣ KORELASI ANTAR SOAL
    # ==================================================
    st.header("5️⃣ Korelasi Antar Soal")

    corr = df[soal_cols].corr()

    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax3)
    ax3.set_title("Heatmap Korelasi Antar Soal")
    st.pyplot(fig3)

    # ==================================================
    # 6️⃣ ANALISIS REGRESI LINEAR
    # ==================================================
    st.header("6️⃣ Analisis Regresi Linear")

    X = df[soal_cols]
    y = df["Total_Nilai"]

    model = LinearRegression()
    model.fit(X, y)

    koef = pd.Series(model.coef_, index=soal_cols)

    fig4, ax4 = plt.subplots(figsize=(10, 4))
    koef.plot(kind="bar", ax=ax4)
    ax4.set_ylabel("Koefisien Regresi")
    ax4.set_title("Pengaruh Soal terhadap Total Nilai")
    st.pyplot(fig4)

    # ==================================================
    # 7️⃣ SEGMENTASI PERFORMA (CLUSTERING)
    # ==================================================
    st.header("7️⃣ Segmentasi Performa Siswa (Clustering)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[["Total_Nilai"]])

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    fig5, ax5 = plt.subplots()
    ax5.scatter(df.index, df["Total_Nilai"], c=df["Cluster"])
    ax5.set_xlabel("Siswa")
    ax5.set_ylabel("Total Nilai")
    ax5.set_title("Cluster Performa Siswa")
    st.pyplot(fig5)

    # ==================================================
    # 8️⃣ TOP 5 SISWA
    # ==================================================
    st.header("8️⃣ Top 5 Siswa")

    top5 = df.sort_values("Total_Nilai", ascending=False).head(5)
    st.dataframe(top5)

else:
    st.info("Silakan upload file Excel untuk memulai dashboard.")