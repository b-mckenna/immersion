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
    if classification == 0:
        os.system('''osascript -e 'display notification "Youre distracted! Get back to work on ''' + user_task + '''" with title "Focus Alert"' ''')
    else:
        os.system('''osascript -e 'display notification "Youre focused! Keep it up on ''' + user_task + '''" with title "Focus Alert"' ''')

def save_classification(frame_bytes, timestamp, classification, latency, user_task, prompt, model):

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
            fieldnames = ['timestamp', 'latency', 'classification', 'user_task', 'model', 'prompt']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if csv_file.tell() == 0:
                writer.writeheader()
            
            writer.writerow({'timestamp': timestamp, 'latency': round(latency), 'classification': classification, 'user_task': user_task, 'model': model, 'prompt': prompt})

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
    prompt = f"Describe what the user is doing by noting open applications and tabs, document or page titles and headers."  #TODO move this to startup config


    timestamp = round(timestamp)
    model = 'llama3.2-vision:11b'#'llama3.2-vision:11b-instruct-q4_K_M' # #llama3.2-vision, llava

    try:
        classification = None 
        print("Calling llama")
        start_time = time.time()
        response = generate(
            model=model,
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

    # call async function to save frame
    save_classification(frame_bytes, timestamp, classification, latency, user_task, prompt, model)

    return classification


def binary_classifier(user_task, llm_response):
    prompt = f"Use the following information to judge whether the user is focused on their task. Reply '1' if they are focused on the task {user_task} and '0' if they are NOT focused on {user_task}. Information: {llm_response}"
    try:
        classification = None 
        print("Calling Llama 3.2 with binary classification prompt")
        start_time = time.time()
        response = generate(
            model='llama3.2',
            prompt=prompt,
            stream=False
        ) 
        classification = response['response']
        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency of llama 3.2: {latency} seconds")
    
    except Exception as e:
        logging.error(f"Ollama API error: {e}")
        raise

    return classification

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
            frame_bytes, image_size = screen_capture()
            print(f"Final image size: {image_size / 1024:.2f} KB")            
            response = classifier(frame_bytes, current_time, user_task)
            binary_classification = binary_classifier(user_task, response)
            status_update(binary_classification, user_task)

            last_capture_time = current_time

if __name__ == '__main__':
    main()