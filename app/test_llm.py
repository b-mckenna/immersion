from ollama import generate
import time
import cv2
import csv
import os
import logging

from test_ui import TestUI

# Create the UI
app = TestUI()

# Update the text
app.update_text("New text from other script!")

# Run the application
#app.run()


prompt = f"Describe what the user is doing in this screenshot."


def save_classification(filename, response, prompt, model):
    try:
        csv_file_path = "output/prompt_testing.csv"
        with open(csv_file_path, mode='a', newline='') as csv_file:
            fieldnames = ['filename', 'response', 'prompt', 'model']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if csv_file.tell() == 0:
                writer.writeheader()
            
            writer.writerow({'filename': filename, 'response': response, 'prompt': prompt, 'model': model})

    except Exception as e:
        logging.error(f"Error saving classification to csv: {e}")
        raise


images_folder = 'output'
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
            print(f"File: {filename}, Response: {response['response']}")
            end_time = time.time()
            latency = end_time - start_time
            print(f"Latency: {latency} seconds")
            save_classification(filename, response['response'], prompt, 'llama3.2-vision:11b-instruct-q4_K_M')
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            time.sleep(2)
            continue