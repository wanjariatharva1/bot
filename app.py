import os
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from io import BytesIO

# Configure Gemini API
genai.configure(api_key="AIzaSyAr7FtFyiQBnk0ih2KX_3rMsixt5ukZtXs")

# FastAPI app instance
app = FastAPI()

# Start the server if running directly
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

# CORS Middleware for frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class PromptRequest(BaseModel):
    prompt: str

# Generate content
@app.post("/api/generate")
async def generate_content(prompt_request: PromptRequest, image: UploadFile = File(None)):
    prompt = prompt_request.prompt
    
    # If image is uploaded, process it here (for simplicity, skipping image processing)
    if image:
        image_bytes = await image.read()
        # Example: image processing or passing to a model

    # Generate content with Gemini API
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return {"output": response.text}
