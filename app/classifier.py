#from openai import OpenAI 
from ollama import generate
from ollama import ChatResponse
import os
import cv2
import csv
import base64
from typing import Dict, List, Optional
import logging

# Configuration
SCREENSHOT_INTERVAL = 10
OPENAI_MODEL = "gpt-4o"
IMAGE_QUALITY = 95

def get_all_mp4_files():
    return [f for f in os.listdir('.') if f.endswith('.mp4')]

def extract_screenshot(video_path: str, time_stamp: int):
    """
    Extract a single frame from a video at a specific timestamp.
    
    Args:
        video_path (str): Path to the video file
        time_stamp (int): Time in seconds where to extract the frame
        
    Returns:
        numpy.ndarray: The extracted frame as an image
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get frames per second
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame number from timestamp
    frame_num = int(time_stamp * fps)
    
    # Set video to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    
    # Read the frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    if ret:
        return frame
    else:
        raise ValueError(f"Could not extract frame at timestamp {time_stamp}")

def extract_screenshots_from_videos():
    """
    Extracts screenshots from all mp4 video files in the current directory at 10-second intervals.

    Returns:
        dict: A dictionary where keys are video filenames and values are lists of extracted frames.
    """
    mp4_files = get_all_mp4_files()  # Get all mp4 files in the current directory
    screenshots = {}  # Initialize a dictionary to store screenshots for each video

    for video in mp4_files:
        screenshots[video] = []  # Initialize a list to store screenshots for the current video
        cap = cv2.VideoCapture(video)  # Open the video file
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get frames per second of the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames in the video
        duration = total_frames / fps  # Calculate the duration of the video in seconds

        for t in range(0, int(duration), 10):  # Loop through the video at 10-second intervals
            try:
                screenshot = extract_screenshot(video, t)  # Extract a screenshot at the current timestamp
                screenshots[video].append(screenshot)  # Append the screenshot to the list for the current video
            except ValueError as e:
                print(e)  # Print an error message if screenshot extraction fails

        cap.release()  # Release the video capture object

    return screenshots  # Return the dictionary of screenshots

def classify_screenshots_openai(screenshots, prompt):
    client = OpenAI(
        organization='org-dyCGXaMGEmhgZNwCEl7C3mSr',
        project='proj_aGN6TnfcDmVMWsGeoSvgRFKa',
    )

    classified_results = {}
    for video, frames in screenshots.items():
        classified_results[video] = []
        for frame in frames:
            # Convert frame to base64 string
            _, buffer = cv2.imencode('.jpg', frame)
            frame_str = base64.b64encode(buffer).decode('utf-8')
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{frame_str}",
                                    }
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                )
                # Update this line to use the correct response structure
                result = response.choices[0].message.content.strip()
                classified_results[video].append(result)
                
            except Exception as e:
                logging.error(f"OpenAI API error: {e}")
                raise
            result = response.choices[0].message.content.strip()
            classified_results[video].append(result)
    
    return classified_results

def extract_binary_classification(classification_str: str) -> Optional[int]:
    """
    Extracts a binary classification (0 or 1) from a given string.

    Args:
        classification_str (str): The string containing the classification.

    Returns:
        Optional[int]: The extracted classification as an integer (0 or 1), or None if not found.
    """
    if '0' in classification_str:
        return "0"
    elif '1' in classification_str:
        return "1"
    else:
        return None

def classify_screenshots_ollama(screenshots, prompt):
    classified_results = {}
    for video, frames in screenshots.items():
        classified_results[video] = []
        for frame in frames:
            # Convert frame to base64 string
            _, buffer = cv2.imencode('.jpg', frame)
            #frame_str = base64.b64encode(buffer).decode('utf-8')
            frame_bytes = buffer.tobytes()

            try:
                result = ""
                for response in generate(
                    model='llama3.2-vision', 
                    prompt=prompt,
                    images=[frame_bytes],
                    stream=True
                ):
                    cleaned_response = extract_binary_classification(response['response'])
                    if cleaned_response is not None:
                        result += cleaned_response
                
                classified_results[video].append(result)

            except Exception as e:
                logging.error(f"Ollama API error: {e}")
                raise
    
    return classified_results

def calculate_distraction_score(classified_results: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Calculate the percentage of time the user was distracted for each video.
    
    Args:
        classified_results: Dictionary of video filenames to list of classifications (0 or 1)
    Returns:
        Dictionary of video filenames to distraction percentages
    """
    distraction_scores = {}
    for video, classifications in classified_results.items():
        # Count number of 0 responses (indicating distraction)
        distractions = sum(1 for result in classifications if result.strip() == "0")
        # Calculate percentage
        distraction_score = (distractions / len(classifications)) * 100 if classifications else 0
        distraction_scores[video] = distraction_score
    return distraction_scores

def main():
    logging.basicConfig(level=logging.INFO)

    #  Setup OpenAI API key is now handled through environment variable
    #if not os.getenv('OPENAI_API_KEY'):
    #    raise ValueError("Please set the OPENAI_API_KEY environment variable")

    # Define the user task
    user_task = "Learning about control flow"
    system_prompt = "You are a bot that classifies whether a screenshot is related to a given task by giving a 1 for yes and 0 for no."
    prompt = f"{system_prompt}\n\n Is the following screenshot related to the task? respond with '1' for yes and '0' for no. DO NOT provide an explanation. For example '1'. Task: {user_task}?"

    # Extract screenshots
    screenshots = extract_screenshots_from_videos()

    # Classify screenshots
    results = classify_screenshots_ollama(screenshots, prompt)

    # Define the CSV file path
    csv_file_path = "classification_results.csv"

    # Write results to CSV
    import uuid
    session_id = str(uuid.uuid4())

    with open(csv_file_path, mode='a', newline='') as csv_file:
        fieldnames = ['session_id', 'video', 'frame_time', 'classification']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        if csv_file.tell() == 0:
            writer.writeheader()
        
        for video, classifications in results.items():
            for i, result in enumerate(classifications):
                writer.writerow({'session_id': session_id, 'video': video, 'frame_time': i * SCREENSHOT_INTERVAL, 'classification': result.strip()})
    
    # Calculate and print distraction scores
    distraction_scores = calculate_distraction_score(results)
    
    # Print detailed results and distraction scores
    for video, classifications in results.items():
        print(f"\nResults for {video}:")
        for i, result in enumerate(classifications):
            print(f"Frame {i*10}s: {result}")
        print(f"Distraction score: {distraction_scores[video]:.1f}% of time distracted")
    
if __name__ == '__main__':
    main()
