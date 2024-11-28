#from openai import OpenAI 
from ollama import generate
import os
import csv
import logging
import numpy as np
import cv2
from mss import mss
import time
import numpy
import argparse


def status_update(classification, user_task):
    import subprocess
    title = "Immersion"
    message = f"You're distracted! Get back to work on {user_task}." if classification == 0 else f"You're focused! Keep it up on {user_task}."
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script])

    print(f"Status updated")

def save_classification(frame_bytes, timestamp, classification, latency, user_task):
    try:
        os.makedirs('output', exist_ok=True)
        filename = f"output/{timestamp}.jpg"
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(filename, frame)
        print(f"Processing frame captured at {timestamp}, saved as {filename}")
    except Exception as e:
        logging.error(f"Error saving frame: {e}")
        raise

    try:
        csv_file_path = "output/run_classifications.csv"
        with open(csv_file_path, mode='a', newline='') as csv_file:
            fieldnames = ['timestamp', 'latency', 'classification', 'user_task']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if csv_file.tell() == 0:
                writer.writeheader()
            
            writer.writerow({'timestamp': timestamp, 'latency': latency, 'classification': classification, 'user_task': user_task})

    except Exception as e:
        logging.error(f"Error saving classification to csv: {e}")
        raise

def screen_capture():
    sct = mss()
    monitor = sct.monitors[1]  # Monitor 1 (primary screen)
    frame = numpy.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    
    return frame_bytes, len(frame_bytes)


def classifier(frame_bytes, timestamp, user_task):
    """
        classify a frame as FOCUS(1) or DISTRACTION(0).
    """
    prompt = f"Is the screenshot a picture of the user working on the task {user_task}? Respond with '1' for yes and '0' for no. DO NOT provide an explanation. For example '1'. " #TODO move this to startup config

    timestamp = round(timestamp)

    try:
        classification = None 
        print("Calling llama")
        start_time = time.time()
        response = generate(
            model='llama3.2-vision:11b-instruct-q4_K_M',  #llama3.2-vision
            prompt=prompt,
            images=[frame_bytes],
            stream=False
        ) 
        classification = response['response']
        print(f"Model response: " +  classification)
        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency: {latency} seconds")
    
    except Exception as e:
        logging.error(f"Ollama API error: {e}")
        raise

    # update status in UI
    status_update(classification, user_task)

    # call async function to save frame
    save_classification(frame_bytes, timestamp, classification, latency, user_task)

    return True
    

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Describe the task the user is working on.')
    parser.add_argument('--task', type=str, default="Coding the Immersion app",
                        help='The task description to be used for classification.')
    args = parser.parse_args()

    user_task = args.task

    capture_interval = 1
    last_capture_time = time.time()
    while True:
        current_time = time.time()
        if current_time - last_capture_time >= capture_interval:
        
            start_capture_time = time.time()
            frame_bytes, image_size = screen_capture()
            print(f"Final image size: {image_size / 1024:.2f} KB")
            end_capture_time = time.time()
            capture_latency = end_capture_time - start_capture_time
            print(f"Screen capture latency: {capture_latency} seconds")
            
            classifier(frame_bytes, current_time, user_task)

            last_capture_time = current_time

if __name__ == '__main__':
    main()