#from openai import OpenAI 
from ollama import generate
import cv2
import logging
import time


logging.basicConfig(level=logging.INFO)


user_task = "There's a woman in a hat"
prompt = f"Is the following statement true? Respond with '1' for yes and '0' for no. DO NOT provide an explanation. For example '1'. Statement: {user_task}"

i = 0
while i < 10:
    # Read image from local directory as cv2 frame
    image_path = 'test_image.jpg'
    frame = cv2.imread(image_path)

    if frame is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert frame to base64 string
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    try:
        start_time = time.time()
        result = ""
        response = generate(
            model='llava', #llama3.2-vision:11b-instruct-q4_K_M
            prompt=prompt,
            images=[frame_bytes],
            stream=False
        )
        result = response['response']
        print(f"Response: {result}")
                
        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency: {latency} seconds")
        i += 1

    except Exception as e:
        logging.error(f"Ollama API error: {e}")
        raise