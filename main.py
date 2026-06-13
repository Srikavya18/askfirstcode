import os
import sys
import uuid
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from dotenv import load_dotenv
# Ensure the backend directory is in python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)
# Load env from root directory if it exists
root_env = os.path.join(os.path.dirname(backend_dir), ".env")
if os.path.exists(root_env):
    load_dotenv(root_env)
else:
    load_dotenv()
# Import DB setup
import database as db_module
from database import get_db, Thread, Message
# Initialize FastAPI
app = FastAPI(title="Universal Memory Chat API")
# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)
# Initialize DB tables
db_module.init_db()
# Pydantic Schemas
class ThreadCreate(BaseModel):
    title: str
    id: Optional[str] = None
class ThreadResponse(BaseModel):
    id: str
    title: str
    created_at: str
    class Config:
        orm_mode = True
class MessageCreate(BaseModel):
    content: str
class MessageResponse(BaseModel):
    id: int
    thread_id: str
    role: str
    content: str
    created_at: str
    class Config:
        orm_mode = True
# Helper: LLM Generation with Universal Memory
def generate_llm_response(
    db: Session,
    current_thread_id: str,
    current_messages: List[Message],
    user_query: str
) -> str:
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    # 1. Retrieve messages from OTHER threads for Universal Memory
    other_threads = db.query(Thread).filter(Thread.id != current_thread_id).all()
    other_threads_context = []
    for t in other_threads:
        t_msgs = db.query(Message).filter(Message.thread_id == t.id).order_by(Message.created_at.desc()).limit(10).all()
        t_msgs.reverse()
        if t_msgs:
            thread_summary = f"Thread: \"{t.title}\" (ID: {t.id})\n"
            for m in t_msgs:
                role_name = "User" if m.role == "user" else "Assistant"
                thread_summary += f"- {role_name}: {m.content}\n"
            other_threads_context.append(thread_summary)
    # Format the universal memory text
    if other_threads_context:
        memory_prompt = (
            "You have access to the conversation history of other threads (Universal Memory). "
            "If the user asks questions about information they mentioned in other threads, "
            "use this context to answer accurately. Cross-reference thread details as needed.\n\n"
            "Here is the history of other threads:\n"
            + "\n---\n".join(other_threads_context)
            + "\n=========================================\n"
        )
    else:
        memory_prompt = "No previous threads exist yet. This is the first conversation thread.\n"
    # Define base system prompt
    system_prompt = (
        "You are an AI assistant designed with Universal Memory.\n"
        "Your goal is to answer the user's latest query using both the current thread's conversation history "
        "and any relevant context from other past threads.\n\n"
        f"{memory_prompt}"
        "Respond naturally. Do not explicitly say 'According to Thread X' unless asked. "
        "Just use the information seamlessly."
    )
    print("\n--- INJECTED SYSTEM PROMPT (UNIVERSAL MEMORY) ---")
    print(system_prompt)
    print("-------------------------------------------------\n")
    current_chat_history = []
    for m in current_messages:
        current_chat_history.append({"role": m.role, "content": m.content})
    current_chat_history.append({"role": "user", "content": user_query})
    # Case A: Gemini API
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                system_instruction=system_prompt
            )
            contents = []
            for msg in current_chat_history:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [msg["content"]]})
            
            response = model.generate_content(contents)
            return response.text
        except Exception as e:
            print(f"Gemini generation error: {e}")
            return f"Error using Gemini API: {str(e)}"
    # Case B: OpenAI API
    elif openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            messages_payload = [{"role": "system", "content": system_prompt}]
            for msg in current_chat_history:
                messages_payload.append({"role": msg["role"], "content": msg["content"]})
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages_payload
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI generation error: {e}")
            return f"Error using OpenAI API: {str(e)}"
    # Case C: Smart Mock Provider
    else:
        lower_query = user_query.lower()
        print("[LLM Backend] Running in Mock Mode. No API Key provided.")
        
        all_text_across_threads = ""
        for summary in other_threads_context:
            all_text_across_threads += summary.lower() + "\n"
        for m in current_messages:
            all_text_across_threads += f" {m.content.lower()}"
        all_text_across_threads += f" {lower_query}"
        # Check for both name and location query
        has_name = "name" in lower_query or "who" in lower_query
        has_loc = "live" in lower_query or "location" in lower_query or "where" in lower_query or "going" in lower_query or "trip" in lower_query
        if has_name and has_loc:
            import re
            name_match = re.search(r"name is ([\w]+)", all_text_across_threads)
            if not name_match:
                name_match = re.search(r"i'm ([\w]+)", all_text_across_threads)
            if not name_match:
                name_match = re.search(r"i am ([\w]+)", all_text_across_threads)
            
            loc_match = re.search(r"live in ([\w]+)", all_text_across_threads)
            if not loc_match:
                loc_match = re.search(r"trip to ([\w]+)", all_text_across_threads)
            if not loc_match:
                loc_match = re.search(r"from ([\w]+)", all_text_across_threads)
            if not loc_match:
                loc_match = re.search(r"going to ([\w]+)", all_text_across_threads)
            name = name_match.group(1).capitalize() if name_match else "John"
            loc = loc_match.group(1).capitalize() if loc_match else "Paris"
            return f"[Mock Mode] Your name is {name} and you are planning a trip to {loc}, based on our chat history."
        if "name" in lower_query or "who" in lower_query:
            import re
            name_match = re.search(r"name is ([\w]+)", all_text_across_threads)
            if not name_match:
                name_match = re.search(r"i'm ([\w]+)", all_text_across_threads)
            if not name_match:
                name_match = re.search(r"i am ([\w]+)", all_text_across_threads)
            
            if name_match:
                name = name_match.group(1).capitalize()
                return f"[Mock Mode] Your name is {name}, which I remember from our conversations."
            return "[Mock Mode] I checked our conversation history, but you haven't mentioned your name yet."
        if "live" in lower_query or "location" in lower_query or "where" in lower_query or "going" in lower_query or "trip" in lower_query:
            import re
            loc_match = re.search(r"live in ([\w]+)", all_text_across_threads)
            if not loc_match:
                loc_match = re.search(r"trip to ([\w]+)", all_text_across_threads)
            if not loc_match:
                loc_match = re.search(r"from ([\w]+)", all_text_across_threads)
            if not loc_match:
                loc_match = re.search(r"going to ([\w]+)", all_text_across_threads)
            
            if loc_match:
                loc = loc_match.group(1).capitalize()
                return f"[Mock Mode] I remember that you are connected with {loc}."
            return "[Mock Mode] I couldn't find details in our previous conversations about where you live or where you are going."
        if "pizza" in lower_query or "eat" in lower_query or "food" in lower_query:
            if "pizza" in all_text_across_threads:
                return "[Mock Mode] I recall from another thread that you love pizza!"
        response_text = (
            "[Mock Mode] API key not found. Universal Memory simulation is active.\n"
            "I checked across all active threads. Here is what I know:\n"
        )
        if other_threads:
            response_text += f"- I see you have {len(other_threads)} other thread(s) active.\n"
            for t in other_threads:
                response_text += f"  * Thread '{t.title}' is active.\n"
        else:
            response_text += "- This is the only thread.\n"
        
        response_text += f"\nYour current message is: '{user_query}'"
        return response_text
# FastAPI API Routes
@app.post("/threads")
def create_thread(thread: ThreadCreate, db: Session = Depends(get_db)):
    t_id = thread.id or str(uuid.uuid4())
    existing = db.query(Thread).filter(Thread.id == t_id).first()
    if existing:
        return {
            "id": existing.id,
            "title": existing.title,
            "created_at": existing.created_at.isoformat()
        }
    
    new_thread = Thread(id=t_id, title=thread.title)
    db.add(new_thread)
    db.commit()
    db.refresh(new_thread)
    return {
        "id": new_thread.id,
        "title": new_thread.title,
        "created_at": new_thread.created_at.isoformat()
    }
@app.get("/threads")
def get_threads(db: Session = Depends(get_db)):
    threads = db.query(Thread).order_by(Thread.created_at.desc()).all()
    return [
        {
            "id": t.id,
            "title": t.title,
            "created_at": t.created_at.isoformat()
        }
        for t in threads
    ]
@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: str, db: Session = Depends(get_db)):
    thread = db.query(Thread).filter(Thread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    db.delete(thread)
    db.commit()
    return {"message": f"Thread {thread_id} deleted successfully"}
@app.get("/threads/{thread_id}/messages")
def get_messages(thread_id: str, db: Session = Depends(get_db)):
    thread = db.query(Thread).filter(Thread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    messages = db.query(Message).filter(Message.thread_id == thread_id).order_by(Message.created_at.asc()).all()
    return [
        {
            "id": m.id,
            "thread_id": m.thread_id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat()
        }
        for m in messages
    ]
@app.post("/threads/{thread_id}/messages")
def send_message(thread_id: str, msg: MessageCreate, db: Session = Depends(get_db)):
    thread = db.query(Thread).filter(Thread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    user_msg = Message(thread_id=thread_id, role="user", content=msg.content)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    current_messages = db.query(Message).filter(Message.thread_id == thread_id).order_by(Message.created_at.asc()).all()
    current_chat_history_excluding_latest = [m for m in current_messages if m.id != user_msg.id]
    assistant_content = generate_llm_response(
        db=db,
        current_thread_id=thread_id,
        current_messages=current_chat_history_excluding_latest,
        user_query=msg.content
    )
    assistant_msg = Message(thread_id=thread_id, role="assistant", content=assistant_content)
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)
    return {
        "user_message": {
            "id": user_msg.id,
            "thread_id": user_msg.thread_id,
            "role": user_msg.role,
            "content": user_msg.content,
            "created_at": user_msg.created_at.isoformat()
        },
        "assistant_message": {
            "id": assistant_msg.id,
            "thread_id": assistant_msg.thread_id,
            "role": assistant_msg.role,
            "content": assistant_msg.content,
            "created_at": assistant_msg.created_at.isoformat()
        }
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
