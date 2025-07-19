import json
from sentence_transformers import SentenceTransformer, util
import streamlit as st

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('dataset.json', 'r',
encoding="utf-8") as file: dataset = json.load(file)

st.title("Chatbot AI Game Roblox / Minecraft")

questions = [item['question'] for
item in dataset]
question_embeddings = model.encode(questions, convert_to_tensor=True)

def chatbot(user_input):
    user_embedding = model.encode(user_input,
convert_to_tensor=True)
    scores = util.cos_sim(user_embedding, question_embeddings)[0]

    best_score = float(scores.max())
    best_idx = int(scores.argmax())

    if best_score > 0.6:
        return dataset[best_idx]['answer']
    else: return "Maaf, saya tak pasti tentang soalan tu"

user_input = st.text_input("Tanya apa-apa pasal minecraft atau roblox:")
    
if st.button("Tanya"):
    if user_input: jawapan = chatbot(user_input)
    st.write("Bot:", jawapan)

    
        