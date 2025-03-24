import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('customer_churn.csv.csv')
    return data

customer = load_data()

# Sidebar for task selection
st.sidebar.title("Task Selection")
st.sidebar.header("Select Tasks From Here",divider='blue')
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
if task=="Code For All Tasks":
    with open("code.txt","r") as file:
        data=file.read()
        st.code(data,"python")
# Main content based on task selection
elif task == "Data Manipulation: Total Male Customers":
    st.header("Total Number of Male Customers")
    total_males = sum(customer['gender'] == "Male")
    st.write(f"Total number of male customers: {total_males}")

elif task == "Data Manipulation: Total DSL Customers":
    st.header("Total Number of Customers with DSL Internet Service")
    total_dsl = sum(customer['InternetService'] == "DSL")
    st.write(f"Total number of customers with DSL: {total_dsl}")

elif task == "Data Manipulation: Female Senior Citizens with Mailed Check":
    st.header("Female Senior Citizens with Mailed Check Payment Method")
    new_customer = customer[(customer['gender'] == 'Female') & (customer['SeniorCitizen'] == 1) & (customer['PaymentMethod'] == 'Mailed check')]
    st.write(new_customer.head())

elif task == "Data Manipulation: Tenure < 10 or Total Charges < 500":
    st.header("Customers with Tenure < 10 or Total Charges < 500")
    new_customer = customer[(customer['tenure'] < 10) | (customer['TotalCharges'] < 500)]
    st.write(new_customer.head())

elif task == "Data Visualization: Churn Distribution":
    st.header("Churn Distribution Pie Chart")
    names = customer["Churn"].value_counts().keys().tolist()
    sizes = customer["Churn"].value_counts().tolist()
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=names, autopct="%0.1f%%")
    st.pyplot(fig)

elif task == "Data Visualization: Internet Service Distribution":
    st.header("Internet Service Distribution Bar Plot")
    fig, ax = plt.subplots()
    ax.bar(customer["InternetService"].value_counts().keys().tolist(), customer["InternetService"].value_counts().tolist(), color='orange')
    ax.set_xlabel('Categories of Internet Service')
    ax.set_ylabel('Count of categories')
    ax.set_title('Distribution of Internet Service')
    st.pyplot(fig)

elif task == "Model Building: Sequential Model with Tenure":
    st.header("Sequential Model with Tenure as Feature")
    x = customer[['tenure']]
    y = customer[['Churn']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    
    model = Sequential()
    model.add(Dense(12, input_dim=1, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test), verbose=0)
    
    st.write("Model Summary")
    model.summary(print_fn=lambda x: st.text(x))
    
    st.write("Accuracy vs Epochs")
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_title('Model Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')
    st.pyplot(fig)
    
    y_pred = model.predict_classes(x_test)
    st.write("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

elif task == "Model Building: Sequential Model with Dropout":
    st.header("Sequential Model with Dropout Layers")
    x = customer[['tenure']]
    y = customer[['Churn']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    
    model = Sequential()
    model.add(Dense(12, input_dim=1, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test), verbose=0)
    
    st.write("Model Summary")
    model.summary(print_fn=lambda x: st.text(x))
    
    st.write("Accuracy vs Epochs")
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_title('Model Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')
    st.pyplot(fig)
    
    y_pred = model.predict_classes(x_test)
    st.write("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))

elif task == "Model Building: Sequential Model with Multiple Features":
    st.header("Sequential Model with Multiple Features")
    x = customer[['MonthlyCharges', 'tenure', 'TotalCharges']]
    y = customer[['Churn']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    
    model = Sequential()
    model.add(Dense(12, input_dim=3, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test), verbose=0)
    
    st.write("Model Summary")
    model.summary(print_fn=lambda x: st.text(x))
    
    st.write("Accuracy vs Epochs")
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_title('Model Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['Train', 'Test'], loc='upper left')
    st.pyplot(fig)
    
    y_pred = model.predict_classes(x_test)
    st.write("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))
