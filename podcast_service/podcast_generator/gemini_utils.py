import os
import google.generativeai as genai

def configure_gemini(api_key):
    genai.configure(api_key=api_key)

def generate_text(prompt, model_name="gemini-2.0-flash-lite"):
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text


