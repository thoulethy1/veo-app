from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
import os

from openai import OpenAI

# Load environment variables
load_dotenv()

# OpenAI client (Hugging Face injects OPENAI_API_KEY automatically)
client = OpenAI()

# FastAPI app
app = FastAPI()

# Middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "change-this-secret"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# Prompt templates
PROMPT_TEMPLATES = {
    "cinematic": (
        "Write a highly cinematic, visually rich video prompt. "
        "Include lighting, camera movement, mood, atmosphere, and dramatic pacing.\n\n"
        "User idea: {prompt}"
    ),
    "realistic": (
        "Write a realistic, natural video prompt as if filmed in real life. "
        "Avoid fantasy elements. Focus on authenticity and subtle details.\n\n"
        "User idea: {prompt}"
    ),
    "anime": (
        "Write an anime-style video prompt. "
        "Include stylized visuals, expressive characters, vibrant colors, and dynamic motion.\n\n"
        "User idea: {prompt}"
    ),
    "documentary": (
        "Write a documentary-style video prompt. "
        "Neutral tone, informative narration, observational camera style.\n\n"
        "User idea: {prompt}"
    ),
}

# Request model
class PromptRequest(BaseModel):
    prompt: str
    mode: str = "cinematic"
    strength: str = "medium"

# Root
@app.get("/")
def home():
    return {"message": "Hello! My app is working 🚀"}

# UI
BASE_DIR = Path(__file__).resolve().parent

@app.get("/ui")
def ui():
    return FileResponse(BASE_DIR / "static" / "index.html")

# Generate endpoint
@app.post("/generate")
def generate(req: PromptRequest, request: Request):
    session = request.session
    last_output = session.get("last_output", "")

    template = PROMPT_TEMPLATES.get(req.mode, PROMPT_TEMPLATES["cinematic"])

    creativity = {
        "soft": "Subtle enhancement, minimal deviation.",
        "wild": "Highly imaginative, bold, dramatic, and unexpected.",
    }.get(req.strength, "Balanced creativity with cinematic clarity.")

    augmented_prompt = ""

    if last_output:
        augmented_prompt += f"""
Previous output (DO NOT repeat wording, structure, or phrases):
{last_output}
"""

    augmented_prompt += f"""
Creativity level:
{creativity}

User prompt:
{req.prompt}

Create a NEW variation.
Change camera angles, pacing, mood, lighting, and composition.
Avoid any overlap with the previous output.
"""

    final_prompt = template.format(prompt=augmented_prompt)

    response = client.responses.create(
        model="gpt-4o-mini",
        input=final_prompt,
    )

    last_output = response.output_text
    session["last_output"] = last_output

    return {"reply": last_output}

# Chat endpoint
@app.get("/ask")
def ask(prompt: str, request: Request):
    session = request.session
    chat_history = session.get("chat_history", [])

    chat_history.append({"role": "user", "content": prompt})

    response = client.responses.create(
        model="gpt-4o-mini",
        input=chat_history,
    )

    ai_message = response.output_text

    chat_history.append({"role": "assistant", "content": ai_message})
    session["chat_history"] = chat_history[-10:]

    return {
        "answer": ai_message,
        "history": chat_history,
    }
