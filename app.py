import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# Konfigurasi Streamlit
st.set_page_config(page_title="Prediksi Rating IMDb", layout="centered")
st.title("🎬 Prediksi Rating Film IMDb (Versi Peningkatan Akurasi)")

# Konversi durasi string ke menit
def convert_duration_to_minutes(duration_str):
    if pd.isna(duration_str):
        return None
    match = re.match(r'(?:(\d+)h)?\s*(?:(\d+)m)?', str(duration_str).strip())
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    return None

# Upload dataset
uploaded_file = st.file_uploader("📂 Upload dataset IMDb (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Data Awal")
    st.dataframe(df.head())

    # Cek kolom
    if 'Duration' in df.columns:
        df['Duration'] = df['Duration'].apply(convert_duration_to_minutes)

    if 'Votes' in df.columns:
        df['Votes_log'] = np.log1p(df['Votes'])

    st.subheader("📌 Informasi Kolom")
    st.write(df.columns.tolist())

    # Fitur numerik yang mungkin
    possible_features = ['Year', 'Votes_log', 'Duration', 'Meta Score']
    available_features = [f for f in possible_features if f in df.columns]
    fitur_terpakai = st.multiselect("✅ Pilih fitur untuk prediksi", available_features, default=available_features)

    # Cek target
    if 'Rating' not in df.columns:
        st.error("❗ Dataset harus memiliki kolom 'Rating'.")
    elif not fitur_terpakai:
        st.warning("⚠️ Pilih minimal satu fitur.")
    else:
        # Dataframe modelling
        df_model = df[fitur_terpakai + ['Rating']].dropna()

        # Outlier removal (IQR)
        Q1 = df_model.quantile(0.25)
        Q3 = df_model.quantile(0.75)
        IQR = Q3 - Q1
        df_model = df_model[~((df_model < (Q1 - 1.5 * IQR)) | (df_model > (Q3 + 1.5 * IQR))).any(axis=1)]

        X = df_model[fitur_terpakai]
        y = df_model['Rating']

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X_scaled)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

        # Model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluasi
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("📈 Evaluasi Model")
        st.write(f"📊 R² Score: **{r2:.3f}**")
        st.write(f"📉 Mean Squared Error: **{mse:.3f}**")

        # Cross-Validation
        cv_scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
        st.write(f"📋 Cross-Validation R² (mean ± std): **{cv_scores.mean():.3f} ± {cv_scores.std():.3f}**")

        # Plot Prediksi vs Aktual
        st.subheader("🎯 Grafik Prediksi vs Aktual")
        fig1, ax1 = plt.subplots()
        ax1.scatter(y_test, y_pred, alpha=0.5, color='blue')
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax1.set_xlabel("Rating Aktual")
        ax1.set_ylabel("Rating Prediksi")
        ax1.set_title("Perbandingan Rating Aktual vs Prediksi")
        st.pyplot(fig1)

        # Distribusi Error
        st.subheader("📊 Distribusi Error")
        errors = y_test - y_pred
        fig2, ax2 = plt.subplots()
        ax2.hist(errors, bins=20, color='orange', edgecolor='black')
        ax2.set_title("Distribusi Error")
        ax2.set_xlabel("Error")
        ax2.set_ylabel("Jumlah")
        st.pyplot(fig2)

        # Korelasi antar fitur asli
        st.subheader("🔗 Korelasi Fitur Asli")
        fig3, ax3 = plt.subplots()
        sns.heatmap(df_model.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax3)
        st.pyplot(fig3)

else:
    st.info("💡 Silakan upload file CSV untuk memulai.")
