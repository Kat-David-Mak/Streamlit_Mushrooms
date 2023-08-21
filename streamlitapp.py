import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

css = """
h1 {
    color: white;
    text-align: center;
}

"""

st.set_page_config(page_title='Mushroom Classification', layout='wide')

# Apply the CSS to the app
st.write(f'<style>{css}</style>', unsafe_allow_html=True)
st.title("Mushroom Streamlit App")
st.header("Welcome!")

# Load the data
mushroom_csv_data = pd.read_csv('end assessment/mushroom_species.csv')

# Create an instance of LabelEncoder
le = LabelEncoder()

# Iterate through each column in the dataframe
for col in mushroom_csv_data.columns:
    # Check if the data type of the column is non-numeric (i.e. categorical)
    if mushroom_csv_data[col].dtype == 'object':
        # Fit and transform the column using LabelEncoder
        mushroom_csv_data[col] = le.fit_transform(mushroom_csv_data[col])

# Print the encoded data
print(mushroom_csv_data)

X = mushroom_csv_data.iloc[:, 0:22].values
Y = mushroom_csv_data.iloc[:, 22:23].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# create a an instance of the Decision Forest Classifier Model
decision_tree = RandomForestClassifier()

# train the Decision Forest Classifier Model
Y_train = np.array(Y_train)
decision_tree.fit(X_train, Y_train.ravel())

# accuracy, confusion matrix, precision score and recall score for Random ForestClassifier Model
y_pred = decision_tree.predict(X_test)
c_matrix = confusion_matrix(Y_test, y_pred)
pre_score = precision_score(Y_test, y_pred)
r_score = recall_score(Y_test, y_pred)

st.write("<h4 style='color: green;'>Random ForestClassifier Model Results</h4>", unsafe_allow_html=True)
st.write('Accuracy:   ', accuracy_score(Y_test, y_pred))
st.write('Precision Score: ', pre_score)
st.write('Recall Score: ', r_score)
st.write('Confusion Matrix: ', c_matrix)