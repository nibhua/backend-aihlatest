import requests
import json

# Test the FastAPI endpoints
BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    response = requests.get(f"{BASE_URL}/")
    print("Root endpoint response:", response.json())

def test_overview_podcast():
    data = {
        "text_input": "Artificial Intelligence is transforming the world. Machine learning algorithms are being used in various industries to automate processes and make predictions. Deep learning, a subset of machine learning, uses neural networks to solve complex problems.",
        "podcast_type": "overview",
        "output_filename": "test_overview.mp3"
    }
    
    response = requests.post(f"{BASE_URL}/generate_podcast", json=data)
    print("Overview podcast response:", response.json())

def test_conversational_podcast():
    data = {
        "text_input": "Climate change is one of the most pressing issues of our time. Rising global temperatures are causing ice caps to melt, sea levels to rise, and weather patterns to become more extreme.",
        "podcast_type": "conversational",
        "output_filename": "test_conversational.mp3"
    }
    
    response = requests.post(f"{BASE_URL}/generate_podcast", json=data)
    print("Conversational podcast response:", response.json())

if __name__ == "__main__":
    print("Testing Podcast Generator API...")
    
    try:
        test_root_endpoint()
        test_overview_podcast()
        test_conversational_podcast()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running on localhost:8000")
    except Exception as e:
        print(f"Error: {e}")

