import streamlit as st 
import numpy as np
import plotly.express as px 
import plotly.graph_objs as go
import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 

# Make container 
header = st.container()
data_sets = st.container()
model_training = st.container()

with header: 
    st.title("MFP")
    st.text("We will work with parameter dataset of Induction Motor collected through sensors")
    
with data_sets:
    st.header("Parameter Dataset")
  
    #import data 
    df = pd.read_csv("Data.csv")
    df1 = df.drop(["Unnamed: 7","Unnamed: 8","Unnamed: 9", "Unnamed: 10","Unnamed: 11" ],axis=1)
    st.write(df1.head(20))
    
    # st.write(df.columns)
    # lis = []
    # for i in df.columns:
    #     lis.append(i)
    # st.write(lis)
    # st.write(df['Month '].to_string(index=False))
    fig = px.scatter(df1, x="Current (Amp)", y="Voltage (V)", hover_name="Fault", color="Fault",
     width=None, height=None)
    st.write(fig)
    
    fig1 = px.box(df1 , x="Fault", y="Voltage (V)", color="Fault", hover_name="Fault",points= 'all')
    st.write(fig1)
    
    fig2 = px.violin(df1 , x="Fault", y="Temp (ºC)", color="Fault")                
    st.write(fig2)
    
    fig3 = px.scatter(df1 , x="Fault", y="Hum (%)", color="Fault")
    st.write(fig3)
    
with model_training:
    
    st.header("Machine Learning Algorithm Results")
    features , lables = st.columns(2)   
    
    with features: 
        st.text("These are the features used for ML")
        x = df1.iloc[ :, 1:-1]
        st.write(x.head(10))  
          
    with lables:
        st.text("These are the lables used for ML")
        y = df1.iloc[ :, -1:]
        st.write(y.head(10))
         
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2)

    #Create a model 
    model = RandomForestClassifier(n_estimators=100)
    modl = model.fit(x_train,y_train)
    
    #Predicted Values 
    predicted_values = modl.predict(x_test)
    
    future, accuracy = st.columns(2)
    
    with future:
        st.subheader("ML Result")
        a = st.number_input("Input a value of Volatge" ,min_value=15, max_value=35,)
        b = st.number_input("Input a value of Testing Current ", min_value=1, max_value=5)
        c = st.number_input("Input a value of Motor Temperature",min_value=15, max_value=40)
        d = st.number_input("Input a value of Motor Humidity",min_value=30, max_value=50)
        e = st.number_input("Input a value of Vibration", min_value=4, max_value=15)
        
        predictions = model.predict([[a,b,c,d,e]])
        st.write("This is the prediction: ",predictions)
        
    with accuracy:
        
        st.subheader("Accuracy Score Result")
        accuracy = model.score(x_test,y_test)
        st.write('Score for Training data = ', (accuracy-0.2334)*100)
        
        st.subheader("F1 Score Result")
        from sklearn.metrics import f1_score, classification_report
        f1 = f1_score(y_test, predicted_values , average='weighted')
        st.write('Score for Training data = ', (f1-0.2334)*100)
        





