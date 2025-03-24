import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    try:
        # Ensure the file path is correct and handle errors
        data = pd.read_csv('customer_churn.csv')  # Ensure the file exists in the same directory
        return data
    except FileNotFoundError:
        st.error("File 'customer_churn.csv' not found. Please upload the file.")
        return None

customer = load_data()

# Sidebar for task selection
st.sidebar.title("Task Selection")
st.sidebar.header("Select Tasks From Here", divider='blue')
task = st.sidebar.radio("Choose a task", [
    "Code For All Tasks",
    "Data Manipulation: Total Male Customers",
    "Data Manipulation: Total DSL Customers",
    "Data Manipulation: Female Senior Citizens with Mailed Check",
    "Data Manipulation: Tenure < 10 or Total Charges < 500",
    "Data Visualization: Churn Distribution",
    "Data Visualization: Internet Service Distribution",
    "Model Building: Sequential Model with Tenure",
    "Model Building: Sequential Model with Dropout",
    "Model Building: Sequential Model with Multiple Features"
])

if task == "Code For All Tasks":
    st.header("Complete Code")
    try:
        with open("code.txt", "r") as file:
            data = file.read()
            st.code(data, "python")
    except FileNotFoundError:
        st.error("File 'code.txt' not found. Please ensure it exists in the same directory.")

# Preprocessing function
def preprocess_data(customer):
    # Drop rows with missing values
    customer = customer.dropna()

    # Convert categorical columns to numeric
    customer['gender'] = customer['gender'].map({'Male': 1, 'Female': 0})
    customer['Partner'] = customer['Partner'].map({'Yes': 1, 'No': 0})
    customer['Dependents'] = customer['Dependents'].map({'Yes': 1, 'No': 0})
    customer['PhoneService'] = customer['PhoneService'].map({'Yes': 1, 'No': 0})
    customer['MultipleLines'] = customer['MultipleLines'].map({'Yes': 1, 'No': 0, 'No phone service': -1})
    customer['InternetService'] = customer['InternetService'].map({'DSL': 1, 'Fiber optic': 2, 'No': 0})
    customer['OnlineSecurity'] = customer['OnlineSecurity'].map({'Yes': 1, 'No': 0, 'No internet service': -1})
    customer['OnlineBackup'] = customer['OnlineBackup'].map({'Yes': 1, 'No': 0, 'No internet service': -1})
    customer['DeviceProtection'] = customer['DeviceProtection'].map({'Yes': 1, 'No': 0, 'No internet service': -1})
    customer['TechSupport'] = customer['TechSupport'].map({'Yes': 1, 'No': 0, 'No internet service': -1})
    customer['StreamingTV'] = customer['StreamingTV'].map({'Yes': 1, 'No': 0, 'No internet service': -1})
    customer['StreamingMovies'] = customer['StreamingMovies'].map({'Yes': 1, 'No': 0, 'No internet service': -1})
    customer['Contract'] = customer['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
    customer['PaperlessBilling'] = customer['PaperlessBilling'].map({'Yes': 1, 'No': 0})
    customer['PaymentMethod'] = customer['PaymentMethod'].map({
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    })
    customer['Churn'] = customer['Churn'].map({'Yes': 1, 'No': 0})

    # Convert TotalCharges to numeric, coercing errors to NaN and dropping them
    customer['TotalCharges'] = pd.to_numeric(customer['TotalCharges'], errors='coerce')
    customer = customer.dropna()

    return customer

# Apply preprocessing
if customer is not None:
    customer = preprocess_data(customer)

# Main content based on task selection
elif task == "Data Manipulation: Total Male Customers":
    st.header("Code: Total Number of Male Customers")
    with st.echo():
        total_males = (customer['gender'] == 1).sum()
        st.write(f"Total number of male customers: {total_males}")

elif task == "Data Manipulation: Total DSL Customers":
    st.header("Code: Total Number of Customers with DSL Internet Service")
    with st.echo():
        total_dsl = (customer['InternetService'] == 1).sum()
        st.write(f"Total number of customers with DSL: {total_dsl}")

elif task == "Data Manipulation: Female Senior Citizens with Mailed Check":
    st.header("Code: Female Senior Citizens with Mailed Check Payment Method")
    with st.echo():
        filtered_customers = customer[
            (customer['gender'] == 0) & 
            (customer['SeniorCitizen'] == 1) & 
            (customer['PaymentMethod'] == 1)
        ]
        st.write(filtered_customers.head())

elif task == "Data Manipulation: Tenure < 10 or Total Charges < 500":
    st.header("Code: Customers with Tenure < 10 or Total Charges < 500")
    with st.echo():
        filtered_customers = customer[
            (customer['tenure'] < 10) | 
            (customer['TotalCharges'] < 500)
        ]
        st.write(filtered_customers.head())

elif task == "Data Visualization: Churn Distribution":
    st.header("Code: Churn Distribution Pie Chart")
    with st.echo():
        churn_counts = customer["Churn"].value_counts()
        names = churn_counts.index.tolist()
        sizes = churn_counts.values.tolist()
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=names, autopct="%0.1f%%")
        st.pyplot(fig)

elif task == "Data Visualization: Internet Service Distribution":
    st.header("Code: Internet Service Distribution Bar Plot")
    with st.echo():
        service_counts = customer["InternetService"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(service_counts.index, service_counts.values, color='orange')
        ax.set_xlabel('Categories of Internet Service')
        ax.set_ylabel('Count of categories')
        ax.set_title('Distribution of Internet Service')
        st.pyplot(fig)

elif task in [
    "Model Building: Sequential Model with Tenure",
    "Model Building: Sequential Model with Dropout",
    "Model Building: Sequential Model with Multiple Features"
]:
    st.header("Download Options")
    st.write("Model building tasks have been removed. You can download the dataset or the notebook file instead.")
    
    # Download dataset
    if st.button("Download Dataset (customer_churn.csv)"):
        try:
            with open("customer_churn.csv", "rb") as file:
                st.download_button(
                    label="Click here to download",
                    data=file,
                    file_name="customer_churn.csv",
                    mime="text/csv"
                )
        except FileNotFoundError:
            st.error("File 'customer_churn.csv' not found. Please ensure it exists in the same directory.")
    
    # Download .ipynb file
    if st.button("Download Notebook (churn.ipynb)"):
        try:
            with open("churn.ipynb", "rb") as file:
                st.download_button(
                    label="Click here to download",
                    data=file,
                    file_name="churn.ipynb",
                    mime="application/x-ipynb+json"
                )
        except FileNotFoundError:
            st.error("File 'churn.ipynb' not found. Please ensure it exists in the same directory.")
