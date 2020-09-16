import streamlit as st
import pandas as pd
import pickle

st.write("""
# Fortal Apartament Price Prediction App
This app predicts **Fortaleza/Ce/Brazil Apartament Prices**!
""")
st.write('---')

# Loads the Boston House Price Dataset
@st.cache
def input_data():
    junho = pd.read_csv('imoveis_junho_clean.csv')
    return junho

imoveis = input_data()
X = imoveis.drop(['price','price/m2'], axis=1)
Y = imoveis.price

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')


def user_input_features():
    data = {}
    for index, value in X.items():
        if index == 'neighborhood':
            s1 = st.sidebar.selectbox('Neighborhood', (sorted(X['neighborhood'].unique())))
            data.update({index: s1})
        elif index == 'area_m2':
            s = st.sidebar.slider('Area (m2)', float(value.min()), float(value.max()), float(value.mean()))
            data.update({index: s})
        else:
            s1 = st.sidebar.selectbox(index, (1,2,3,4,5))
            data.update({index: s1})
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
df = pd.concat([input_df,X],axis=0)


# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering


dummy = pd.get_dummies(df['neighborhood'])
df = pd.concat([df,dummy], axis=1)
del df['neighborhood']
df= df[:1] # Selects only the first row (the user input data)

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
load_rfr = pickle.load(open('imoveis_rfr.pkl', 'rb'))

# Apply model to make predictions
prediction = load_rfr.predict(df)
#prediction_proba = load_rfr.predict_proba(df)


st.header('Prediction of Price in R$')
st.write(prediction)
st.write('---')
st.write('Obs: This App was created for educational purposes only, prediction prices may not reflect real market prices!')

