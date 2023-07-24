import joblib
import streamlit as st
from nltk.stem.porter import PorterStemmer
import numpy as np
import re
from nltk.corpus import stopwords
def remove(text,pattern) :
    port_stem=PorterStemmer()
    r=re.findall(pattern,text)
    for i in r :
        text=re.sub(i," ",text)
    text=re.sub("[^a-zA-Z#]"," ",text)
    text=text.split()
    text=[port_stem.stem(word) for word in text if not word in stopwords.words("english")]
    text=" ".join(text)
    return text
def log_model (text):
    Log_reg=joblib.load("logisticregressionmodel.sav")
    tfidf=joblib.load("TFIDF.sav")
    transformed_text=np.vectorize(remove)(text,"@[\w]*")
    transformed_text=tfidf.fit_transform(text)
    pred=Log_reg.predict_proba(transformed_text)
    pred_int=pred[:,1]>=0.3
    pred_int=pred_int.astype(int)
    if pred_int== 1 :
        st.write("Tweet is Racist/Sexist")
    else :
        st.write("Normal Tweet")
st.title("Sentiment Analyser")
st.markdown("ML model to analyse tweets and determine the sentiment")
text=st.text_input("Tweet")
if st.button("Predict") :
	sentiment=log_model(text)
