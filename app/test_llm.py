from ollama import generate
import time
import cv2

import os

prompt = f"Is the following statement true? Respond with '1' for yes and '0' for no. DO NOT provide an explanation. For example '1'. Statement: {user_task}"

images_folder = 'images'
for filename in os.listdir(images_folder):
    if filename.endswith('.jpg'):
        try:
            image_path = os.path.join(images_folder, filename)
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            start_time = time.time()
            response = generate(
                model='llama3.2-vision:11b-instruct-q4_K_M',
                prompt=prompt,
                images=[frame_bytes],
                stream=False,
            ) 
            print(f"File: {filename}, Response: {response['message']['content']}")
            end_time = time.time()
            latency = end_time - start_time
            print(f"Latency: {latency} seconds")
            
            # Add a small delay between requests
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            time.sleep(2)
            continue