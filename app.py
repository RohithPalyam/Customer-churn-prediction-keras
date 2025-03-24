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
        
        # Handle missing or invalid values in 'TotalCharges'
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'].fillna(0, inplace=True)
        
        return data
    except FileNotFoundError:
        st.error("File 'customer_churn.csv' not found. Please upload the file.")
        return None

customer = load_data()

if customer is None:
    st.stop()

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

# Main content based on task selection
if task == "Code For All Tasks":
    st.header("Complete Code")
    try:
        with open("code.txt", "r") as file:
            data = file.read()
            st.code(data, "python")
    except FileNotFoundError:
        st.error("File 'code.txt' not found. Please ensure it exists in the same directory.")

elif task == "Data Manipulation: Total Male Customers":
    st.header("Code: Total Number of Male Customers")
    with st.echo():
        total_males = (customer['gender'] == "Male").sum()
        st.write(f"Total number of male customers: {total_males}")

elif task == "Data Manipulation: Total DSL Customers":
    st.header("Code: Total Number of Customers with DSL Internet Service")
    with st.echo():
        total_dsl = (customer['InternetService'] == "DSL").sum()
        st.write(f"Total number of customers with DSL: {total_dsl}")

elif task == "Data Manipulation: Female Senior Citizens with Mailed Check":
    st.header("Code: Female Senior Citizens with Mailed Check Payment Method")
    with st.echo():
        filtered_customers = customer[
            (customer['gender'] == 'Female') & 
            (customer['SeniorCitizen'] == 1) & 
            (customer['PaymentMethod'] == 'Mailed check')
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

elif task == "Model Building: Sequential Model with Tenure":
    st.header("Code: Sequential Model with Tenure as Feature")
    with st.echo():
        x = customer[['tenure']].values.astype('float32')
        y = customer['Churn'].map({'Yes': 1, 'No': 0}).values.astype('int32')

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

        model = Sequential([
            Dense(12, input_dim=1, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test), verbose=0)

        st.write("Model Summary")
        model.summary(print_fn=lambda x: st.text(x))

        st.write("Accuracy vs Epochs")
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Train')
        ax.plot(history.history['val_accuracy'], label='Test')
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper left')
        st.pyplot(fig)

        y_pred = (model.predict(x_test) > 0.5).astype(int)
        st.write("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

elif task == "Model Building: Sequential Model with Dropout":
    st.header("Code: Sequential Model with Dropout Layers")
    with st.echo():
        x = customer[['tenure']].values.astype('float32')
        y = customer['Churn'].map({'Yes': 1, 'No': 0}).values.astype('int32')

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

        model = Sequential([
            Dense(12, input_dim=1, activation='relu'),
            Dropout(0.3),
            Dense(8, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test), verbose=0)

        st.write("Model Summary")
        model.summary(print_fn=lambda x: st.text(x))

        st.write("Accuracy vs Epochs")
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Train')
        ax.plot(history.history['val_accuracy'], label='Test')
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper left')
        st.pyplot(fig)

        y_pred = (model.predict(x_test) > 0.5).astype(int)
        st.write("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

elif task == "Model Building: Sequential Model with Multiple Features":
    st.header("Code: Sequential Model with Multiple Features")
    with st.echo():
        x = customer[['MonthlyCharges', 'tenure', 'TotalCharges']].values.astype('float32')
        y = customer['Churn'].map({'Yes': 1, 'No': 0}).values.astype('int32')

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

        model = Sequential([
            Dense(12, input_dim=3, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test), verbose=0)

        st.write("Model Summary")
        model.summary(print_fn=lambda x: st.text(x))

        st.write("Accuracy vs Epochs")
        fig, ax = plt.subplots()
        ax.plot(history.history['accuracy'], label='Train')
        ax.plot(history.history['val_accuracy'], label='Test')
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper left')
        st.pyplot(fig)

        y_pred = (model.predict(x_test) > 0.5).astype(int)
        st.write("Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))
