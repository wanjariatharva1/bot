from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import google.generativeai as genai
from PIL import Image
import base64

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Generative AI model configuration
genai.configure(api_key="YOUR_API_KEY")  # Replace with your actual API key
model = genai.GenerativeModel('gemini-1.5-flash')

@app.post("/api/drawing")
async def process_drawing(
    image: UploadFile = File(...),  # Drawing canvas image sent as a file
    prompt: str = Form(None),      # Optional user prompt from the frontend
):
    # Read the uploaded image
    img = np.frombuffer(await image.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image data."})

    # Convert the image to a format suitable for AI processing
    pil_image = Image.fromarray(img)
    pil_image = pil_image.resize((640, 480))  # Resize for AI model compatibility

    try:
        # Use the user-provided prompt or a default prompt
        prompt = prompt if prompt else "Default AI prompt for your drawing."
        response = model.generate_content([prompt, pil_image])  # Send to AI model
        ai_output = response.text  # Get the AI output text
    except Exception as e:
        print(f"Error while generating AI content: {str(e)}")
        return JSONResponse(status_code=500, content={"error": "AI processing failed."})

    # Encode the processed image back to base64 for frontend rendering
    _, buffer = cv2.imencode('.jpg', img)
    canvas_b64 = base64.b64encode(buffer).decode('utf-8')

    # Return the AI output text and the canvas image
    return JSONResponse(content={"outputText": ai_output, "canvas": canvas_b64})
