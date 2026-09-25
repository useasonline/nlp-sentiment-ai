import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="24/7 Live NLP AI", page_icon="🧠", layout="centered")

st.title("🧠 Live NLP Sentiment AI")
st.write("This AI model is running 24/7 in the cloud. Type text below to analyze it in real time.")

# 1. Dataset
training_data = [
    ("I love this product, it is amazing", "Positive"),
    ("Great experience, super happy with the quality", "Positive"),
    ("Fantastic work, truly outstanding and helpful", "Positive"),
    ("Best purchase I have ever made, wonderful", "Positive"),
    ("Awesome customer service and fast delivery", "Positive"),
    ("Terrible quality, broke on the first day", "Negative"),
    ("Worst experience ever, I hate it", "Negative"),
    ("Very disappointing and completely useless", "Negative"),
    ("Horrible customer support, total waste of money", "Negative"),
    ("Do not buy this, it is awful and bad", "Negative"),
]

texts = [item[0] for item in training_data]
labels = [item[1] for item in training_data]

# 2. Train Model
@st.cache_resource
def train_model():
    vectorizer = CountVectorizer(lowercase=True)
    X = vectorizer.fit_transform(texts)
    model = MultinomialNB()
    model.fit(X, labels)
    return vectorizer, model

vectorizer, model = train_model()

# 3. Interactive UI
user_text = st.text_area("Enter a sentence to test:", "The service was super fast and awesome!")

if st.button("Predict Sentiment"):
    if user_text.strip():
        vec = vectorizer.transform([user_text])
        prediction = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        classes = model.classes_

        if prediction == "Positive":
            st.success("Result: Positive 😊")
        else:
            st.error("Result: Negative 😞")

        st.subheader("Confidence Scores:")
        for cls, prob in zip(classes, probs):
            st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")
    else:
        st.warning("Please type some words first.")