import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Fungsi untuk memuat data
@st.cache(allow_output_mutation=True)
def load_data():
    # Pastikan untuk mengganti jalur dengan jalur ke file Anda
    data = pd.read_csv('data/CCDATA.csv')  # Ganti dengan path dataset Anda
    data.fillna(0, inplace=True)  # Mengganti NaN dengan 0
    return data

# Mempersiapkan data
data = load_data()

# Normalisasi data (menghindari fitur dengan rentang yang berbeda)
scaler = StandardScaler()
features = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PURCHASES_FREQUENCY', 'AGE']
data_scaled = scaler.fit_transform(data[features])

# 1. Clustering dengan KMeans
num_clusters = st.sidebar.slider('Jumlah Cluster', 2, 10, 5)
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# 2. Probabilitas dengan GMM
gmm = GaussianMixture(n_components=num_clusters, random_state=0)
gmm.fit(data_scaled)
data['Probability'] = gmm.predict_proba(data_scaled).max(axis=1)  # Mengambil probabilitas tertinggi

# 3. Filtering dengan Decision Tree
# Membagi data untuk training dan testing
X = data[features]
y = data['Cluster']  # Target adalah cluster yang dihasilkan dari KMeans

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Membuat model Decision Tree
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

# Memprediksi dan mengevaluasi model
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Menampilkan hasil akurasi
st.write(f"Akurasi Decision Tree: {accuracy:.2f}")

# Menu untuk filtering
st.sidebar.title("Filter Data")
age_filter = st.sidebar.slider('Umur', min_value=18, max_value=100, value=(18, 30))
purchase_filter = st.sidebar.slider('Pengeluaran Total', min_value=0, max_value=10000, value=(0, 1000))
credit_limit_filter = st.sidebar.slider('Limit Kredit', min_value=0, max_value=50000, value=(0, 10000))
transaction_frequency_filter = st.sidebar.slider('Frekuensi Transaksi', min_value=0, max_value=100, value=(0, 10))

# Filter data berdasarkan input pengguna
filtered_data = data[
    (data['AGE'].between(age_filter[0], age_filter[1])) &
    (data['PURCHASES'].between(purchase_filter[0], purchase_filter[1])) &
    (data['CREDIT_LIMIT'].between(credit_limit_filter[0], credit_limit_filter[1])) &
    (data['PURCHASES_FREQUENCY'].between(transaction_frequency_filter[0], transaction_frequency_filter[1]))
]

# Menampilkan data yang telah difilter
st.write("Data yang difilter:")
st.dataframe(filtered_data)

# Visualisasi Cluster
plt.figure(figsize=(10, 6))
plt.scatter(data['BALANCE'], data['PURCHASES'], c=data['Cluster'], cmap='viridis')
plt.title('Cluster Pelanggan')
plt.xlabel('Saldo')
plt.ylabel('Pengeluaran')
plt.colorbar(label='Cluster')
st.pyplot(plt)

# Menampilkan probabilitas
st.write("Probabilitas keanggotaan cluster untuk setiap pelanggan:")
st.dataframe(data[['CUST_ID', 'Cluster', 'Probability']])
