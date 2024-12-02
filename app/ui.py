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
from tkinter import *
from tkinter import ttk

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
    #save_classification(frame_bytes, timestamp, classification, latency, user_task)

    return True

class ClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Realtime Classifier")
        
        # Configure the main window
        self.root.geometry("400x200")
        
        # Create and pack widgets
        self.status_frame = ttk.Frame(root, padding="10")
        self.status_frame.pack(fill=BOTH, expand=True)
        
        self.status_label = ttk.Label(
            self.status_frame, 
            text="Waiting for classification...",
            font=("Arial", 12)
        )
        self.status_label.pack(pady=20)
        
        self.task_label = ttk.Label(
            self.status_frame,
            text="Current task: Not set",
            font=("Arial", 10)
        )
        self.task_label.pack(pady=10)
        
        # Store the user_task
        self.user_task = None
        
    def update_status(self, classification, user_task):
        status_text = "FOCUSED ✅" if classification == "1" else "DISTRACTED ❌"
        self.status_label.config(text=f"Status: {status_text}")
        self.task_label.config(text=f"Current task: {user_task}")
        self.root.update()

# Modify the status_update function to use the GUI
def status_update(classification, user_task):
    if hasattr(main, 'app'):
        main.app.update_status(classification, user_task)

# Modify the main function
def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Describe the task the user is working on.')
    parser.add_argument('--task', type=str, default="Coding the Immersion app",
                        help='The task description to be used for classification.')
    args = parser.parse_args()

    user_task = args.task

    # Create the Tkinter window
    root = Tk()
    main.app = ClassifierApp(root)
    main.app.user_task = user_task

    capture_interval = 1
    last_capture_time = time.time()

    def update_loop():
        nonlocal last_capture_time
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
            
        root.after(100, update_loop)  # Schedule the next check

    update_loop()
    root.mainloop()