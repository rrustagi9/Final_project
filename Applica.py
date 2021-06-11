
import pickle
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.summarization.summarizer import summarize
import pandas as pd



ps = PorterStemmer()

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def model_catch():
    encoder=pickle.load(open('vectorizer_text.pickle','rb'))
    classifier=pickle.load(open('model_MNB_text.pickle','rb'))
    return encoder,classifier
   

def preprocess(para):
    import re
    nltk.download('stopwords')
    from nltk.stem.porter import PorterStemmer
    ps = PorterStemmer()
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', para)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    return corpus

def transform(corpus,encoder):
    X=encoder.transform(corpus).toarray()
    return X


def analyze(headline,encoder,classifier):
    corpus = preprocess(headline)
    x=transform(corpus,encoder)
    pred=classifier.predict(x)
    prob=classifier.predict_proba(x)
    return(pred,prob)


    




st.title('The News Solution')
st.write('Your one step news verification and summarization platform ')
if st.sidebar.button('About Developer'):
        st.sidebar.write('Hi my name is Rahul Rustagi and I am a university of Michigan certified data science professional . Most of my projects and training have been centered around the concept of automation and improved system usability. Not only do I have experience in building and optimization of machine learning and deep learning models ,I also have experience in their front end development and cloud deployment on services like Azure ,GCP and AWS.  ')
        st.sidebar.write('Linkedin profile - https://www.linkedin.com/in/rahul-rustagi-4a1836133/ ')
        st.sidebar.write('Github profile - https://github.com/rrustagi9 ')
if st.sidebar.button('About Project'):
        st.sidebar.write('The project is built using Machine Learning and NLP concepts to provide the service of news classification and summarization to the user ')
if st.sidebar.button('Classification Approach'):
        st.sidebar.write('To predict the probability of a news article being real and not artificially generated I was able to use TF-IDF NLP approach to preprocess the text dataset and to classify the data MUltinomial Naive Bayes algorithm was used to classify the data')

option = st.selectbox(' Please choose the operation you want to perform -',('None', 'Summarize', 'Classify'))
if option=='Classify':
    ption = st.selectbox(' Please choose the approach you want to perform -', ('News Article', 'Headline'))
    text_1=st.text_area('enter the news article - ')
    encoder,classifier=model_catch()
    if st.button(label='analyze'):
        result,prob=analyze(text_1,encoder,classifier)
        if result<=0.5:
            st.write('Based on the semantic structure and word associations the text has higher chances of being fake')
        else:
            st.write('Based on the semantic structure and word associations the text has higher chances of being True')
        st.write('The probability distribution is as follows - ')
        probi=pd.DataFrame(columns=('Result','Probability'))
        probi['Result']=['fake','real']
        probi['Probability']=[prob[0][0],prob[0][1]]
        st.write(probi,index=False)
        #st.write('Probability of article being Fake - ',round(probi['fake']*100),'%')
        #st.write('Probability of article being Real - ',round(probi['real']*100),'%')
        #fig=probi.iplot(x='Result',kind='pie')
        #st.pyplot(fig)
elif option=='Summarize':

    text_1=st.text_area('enter the news article - ')
    ratio=st.slider('enter the summarization ratio',min_value=0.3,max_value=0.8,step=0.1)
    if st.button('Summarize'):
        summ_per= summarize(text_1, ratio = ratio)
        st.write('The summary is - ')
        st.write(summ_per)
    
    
    




