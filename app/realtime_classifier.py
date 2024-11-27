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


def warmup():
    print("Warming up model...")
    # Read image from local directory as cv2 frame
    image_path = 'test_image.jpg'
    frame = cv2.imread(image_path)

    if frame is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert frame to base64 string
    _, buffer = cv2.imencode('.jpg', frame)
    #frame_str = base64.b64encode(buffer).decode('utf-8')
    frame_bytes = buffer.tobytes()

    user_task = "There's a woman in a hat"
    prompt = f"Is the following statement true? Respond with '1' for yes and '0' for no. DO NOT provide an explanation. For example '1'. Statement: {user_task}"

    try:
        start_time = time.time()
        result = ""
        response = generate(
            model='llama3.2-vision:11b-instruct-q4_K_M', 
            prompt=prompt,
            images=[frame_bytes],
            stream=False,
            keep_alive=30
        )
        result = response['response']
        print(f"Response: {result}")
                
        end_time = time.time()
        latency = end_time - start_time
        print(f"Latency: {latency} seconds")

    except Exception as e:
        logging.error(f"Ollama API error: {e}")
        raise

def status_update(classification):
    print(f"Classification: {classification}")

def save_classification(frame, timestamp, classification):
    try:
        os.makedirs('output', exist_ok=True)
        filename = f"output/{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Processing frame captured at {timestamp}, saved as {filename}")
    except Exception as e:
        logging.error(f"Error saving frame: {e}")
        raise

    try:
        csv_file_path = "output/run_classifications.csv"
        with open(csv_file_path, mode='a', newline='') as csv_file:
            fieldnames = ['timestamp', 'classification']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            if csv_file.tell() == 0:
                writer.writeheader()
            
            writer.writerow({'timestamp': timestamp, 'classification': classification})

    except Exception as e:
        logging.error(f"Error saving classification to csv: {e}")
        raise

def classifier(frame, timestamp):
    """
        classify a frame as FOCUS(1) or DISTRACTION(0).
    """
    user_task = "The user is working on something related to software development"
    prompt = f"Is the following statement true? Respond with '1' for yes and '0' for no. DO NOT provide an explanation. For example '1'. Statement: {user_task}" #TODO move this to startup config

    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()

    timestamp = round(timestamp)

    try:
        classification = None 
        print("Calling llama")
        start_time = time.time()
        response = generate(
            model='llama3.2-vision:11b-instruct-q4_K_M',  #llama3.2-vision
            prompt=prompt,
            images=[frame_bytes],
            stream=False,
            #keep_alive=30
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
    status_update(classification)

    # call async function to save frame
    save_classification(frame, timestamp, classification)

    return True
    

def main():
    logging.basicConfig(level=logging.INFO)
    
    warmup() # load model

    sct = mss()
    monitor = sct.monitors[1]  # monitor 1
    last_capture_time = time.time()

    try:
        capture_interval = 10
        while True:
            current_time = time.time()
            if current_time - last_capture_time >= capture_interval:
                frame = numpy.array(sct.grab(monitor))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                #image_path = 'output/1732663045.jpg'
                #frame = cv2.imread(image_path)
                print(f"Frame captured at {current_time}")
                classifier(frame, current_time)

                last_capture_time = current_time

    finally:
        print("Exiting...")

if __name__ == '__main__':
    main()