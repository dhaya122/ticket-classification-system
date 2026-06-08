import streamlit as st
import joblib
from google import genai


model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


def predict_queue(ticket):
    vec = vectorizer.transform([ticket])
    return model.predict(vec)[0]


def generate_reply(ticket, queue):
    prompt = f"""
You are a professional customer support agent.

Ticket:
{ticket}

Category:
{queue}

Write a polite, helpful response.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


st.title("📩 Ticket Classification System")

ticket = st.text_area("Enter Ticket")

if st.button("Predict & Generate Reply"):

    if not ticket.strip():
        st.warning("Enter a ticket first")
        st.stop()

    queue = predict_queue(ticket)
    reply = generate_reply(ticket, queue)

    st.success(f"Predicted Category: {queue}")
    st.subheader("AI Response")
    st.write(reply)