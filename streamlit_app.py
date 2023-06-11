# Library import
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st

# Data import
df = pd.read_csv('Clicked Ads Dataset.csv')
df.head()

# Data Cleansing
df = df.rename(columns= {'Male':'Gender'})
col11, col12, col13, col14 = st.columns(4)
with col11:
    age = st.slider('Age', 0, 130, 25)
with col12:
    income = st.slider("Area Income", 0, 1000000000, 0)
with col13:
    internet_usage = st.slider("Daily Internet Usage", 0, 500, 0)
with col14:
    spent_time = st.slider("Daily Time Spent on Site", 0, 500,  )


# 'Age',
#  'Area Income',
#  'Bank',
#  'Daily Internet Usage',
#  'Daily Time Spent on Site',
#  'Electronic',
#  'Fashion',
#  'Finance',
#  'Food',
#  'Furniture',
#  'Gender',
#  'Health',
#  'House',
#  'Otomotif',
#  'Travel'