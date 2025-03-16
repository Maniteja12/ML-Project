import streamlit as st
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
porter = PorterStemmer()

#We are using the same data_preprocessing code that we had mentioned in the report
def data_preprocessing(message):
    # We are converting message into lowercase
    message = message.lower()
    # We are tokenizing the message into array of words
    data = []

    for i in message:
        # the below line removes any characters from i that are not alphanumeric
        result = re.sub(r'[^a-zA-Z0-9\s]', '', i)
        if (result != ""):
            data.append(result)

    text = data[:]
    data.clear()

    for j in text:
        # For each value in text, we will check for stopwords and special characters.
        if j not in stopwords.words('english') and j not in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
            data.append(j)

    text = data[:]
    data.clear()

    # We are performing stemming
    for k in text:
        result = porter.stem(k)
        if (result):
            data.append(result)

    return " ".join(data)


def main():
    #We are importying both ifidf module and the trained voting model.
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Adding the title to the webpage
    st.title("Spam message classifier")

    #Adding input box to the webpage
    sms_input = st.text_input("Please enter the message you recieved")

    #Inserting button onto the webpage
    if st.button("Predict"):
        transformed_sms = data_preprocessing(sms_input)

        # Transforming the input using the fitted vectorizer
        vector_input = tfidf.transform([transformed_sms])

        # Predict using the loaded model
        result = model.predict(vector_input)[0]

        if result == 1:
            st.header("Spam")
        else:
            st.header("Ham")

if __name__ == "__main__":
    main()



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
