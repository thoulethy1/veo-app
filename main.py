from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 🧠 MEMORY (chat history)
chat_history = [
    {"role": "system", "content": "You are a helpful, friendly AI assistant."}
]

@app.get("/ask")
def ask(prompt: str):
    try:
        chat_history.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=chat_history,
            temperature=0.7
        )

        ai_message = response.choices[0].message.content
        chat_history.append({"role": "assistant", "content": ai_message})

        return {
            "answer": ai_message,
            "history": chat_history
        }

    except Exception as e:
        return {
            "answer": f"Error: {str(e)}"
        }