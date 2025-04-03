import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import seaborn as sns
data_path = "bank_transactions_data_2.csv"
data = pd.read_csv(data_path)
st.set_page_config(page_title="Fraud Transaction Detection")
st.title("Sample data:")
st.dataframe(data.head())
numeric_features = ["TransactionAmount", "CustomerAge", "TransactionDuration", "LoginAttempts", "AccountBalance"]
missing_features = [feature for feature in numeric_features if feature not in data.columns]
data_numeric = data[numeric_features]
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(data_scaled)
data["Cluster"] = clusters

# DBSCAN Clustering
eps = st.sidebar.slider("DBSCAN eps", 0.1, 2.0, 0.5, 0.1)
min_samples = st.sidebar.slider("DBSCAN min_samples", 1, 10, 5)
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_clusters = dbscan.fit_predict(data_scaled)
data["DBSCAN_Cluster"] = dbscan_clusters

data['Fraud'] = (data['Cluster'] == 1).astype(int) 
features = data[numeric_features]
target = data['Fraud']
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(features, target)
knn_model = KNeighborsClassifier()
knn_model.fit(features, target)
def plot_clusters(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x="TransactionAmount", y="AccountBalance", hue="Cluster", palette="viridis")
    plt.title("Cluster Visualization")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Account Balance")
    st.pyplot(plt)
def plot_feature_distribution(data):
    for feature in numeric_features:
        plt.figure(figsize=(10, 4))
        sns.histplot(data[feature], bins=30, kde=True)
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        st.pyplot(plt)
def main():
    st.title("Fraud Transaction Detection App")
    option = st.sidebar.selectbox("Select an option", ["View Dataset", "Predict Fraud Transaction", "Visualize Data"])  
    if option == "View Dataset":
        st.subheader("Dataset")
        st.dataframe(data)    
    elif option == "Predict Fraud Transaction":
        st.subheader("Predict Fraud Transaction")
        model_option = st.sidebar.selectbox("Select Model", ["K-Means", "DBSCAN", "Random Forest", "KNN"])
        st.write("Adjust the values below to predict if a transaction is fraudulent.")
        input_values = {}
        for feature in numeric_features:
            input_values[feature] = st.number_input(feature, value=0.0)      
        if st.button("Predict"):
            input_df = pd.DataFrame([input_values])
            input_scaled = scaler.transform(input_df)          
            if model_option == "K-Means":
                predicted_cluster = kmeans.predict(input_scaled)
                if predicted_cluster[0] == 1:
                    st.warning("This transaction is predicted to be a Fraud Transaction.")
                else:
                    st.success("This transaction is predicted to be Not a Fraud Transaction.")          
            elif model_option == "Random Forest":
                predicted_rf = rf_model.predict(input_df)
                if predicted_rf[0] == 1:
                    st.warning("This transaction is predicted to be a Fraud Transaction.")
                else:
                    st.success("This transaction is predicted to be Not a Fraud Transaction.")        
            elif model_option == "DBSCAN":
                predicted_dbscan = dbscan.fit_predict(input_scaled)
                if predicted_dbscan[0] == -1:
                    st.warning("This transaction is predicted to be Noise (not a Fraud Transaction).")
                else:
                    st.success("This transaction is predicted to be a Fraud Transaction.")
            elif model_option == "KNN":
                predicted_knn = knn_model.predict(input_df)
                if predicted_knn[0] == 1:
                    st.warning("This transaction is predicted to be a Fraud Transaction.")
                else:
                    st.success("This transaction is predicted to be Not a Fraud Transaction.")

    elif option == "Visualize Data":
        st.subheader("Data Visualizations")
        plot_clusters(data)
        plot_feature_distribution(data)
if __name__ == "__main__":
    main()
st.set_page_config(page_title="My App", page_icon="🎵", layout="wide")

# Hide Streamlit's default menu and footer
hide_menu = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

