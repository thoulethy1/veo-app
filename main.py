from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
import os
from openai import OpenAI



load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

client = OpenAI(api_key=OPENAI_API_KEY)



app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET", "change-this-secret"),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Deta requires wildcard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health")
def health():
    return {"status": "ok"}



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



class PromptRequest(BaseModel):
    prompt: str
    mode: str = "cinematic"
    strength: str = "medium"



@app.get("/")
def home():
    return {"message": "Hello! My app is working 🚀"}

BASE_DIR = Path(__file__).resolve().parent

@app.get("/ui")
def ui():
    return FileResponse(BASE_DIR / "static" / "index.html")

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

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.8,
    )

    last_output = response.choices[0].message.content
    session["last_output"] = last_output

    return {"reply": last_output}

@app.get("/ask")
def ask(prompt: str, request: Request):
    session = request.session
    chat_history = session.get("chat_history", [])

    chat_history.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=chat_history,
        temperature=0.7,
    )

    ai_message = response.choices[0].message.content

    chat_history.append({"role": "assistant", "content": ai_message})
    session["chat_history"] = chat_history[-10:]

    return {"answer": ai_message, "history": chat_history}
