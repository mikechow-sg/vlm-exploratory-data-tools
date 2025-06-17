# This script loops through a folder of shot thumbnails from videos as split in 4CAT (Peeters and Hagen 2022) using
# PySceneDetect (Castellano 2025).
# The script then connects with the local LM Studio server and passes a VLM with the prompt and the image.
# The dataframe containing the scene metadata (such as duration etc) is then annotated with the output

import pandas as pd
import numpy as np
import lmstudio as lms
import os
import re
import time
import datetime
import tracemalloc


def main():

    # Establish Directory
    frames_folder = '' # Folder containing thumbnails
    file_path = '' # Scene metadata path
    global log_file # Text log file
    log_file = '' # Text log file (optional)
    log_data_path = '' # CSV log file (optional)

    df_scene_spox = pd.read_csv(file_path)
    df_log = pd.read_csv(log_data_path)
    file_name = os.path.basename(file_path)
    print(f"Gathered data from {file_name}...")

    confirm = 'n'
    while confirm != 'y':
        # prompt = input("Please input the prompt: \n") # Update prompt here
        prompt = "You are an expert annotator of social media videos. You are provided a still image from a scene in a video posted on the Israel Defence Force’s TikTok between 7 October 2023 to 7 October 2024. Your analysis of the setting of the image will enable understanding of their tendencies in their video production processes. Your job is to analyze the backdrop of the image and classify it into the following three mutually exclusive location labels. Ignore overlaid text or captions that may inform your classification. For the provided image, select the one category that best describes its setting: Graphics: Image is not live‑action footage or is artificial (e.g., animations, CGI, black screens) and hence cannot be discernably set in a location; Outdoor: Image is set outdoors with natural lighting and where natural landscapes or exteriors of urban environments may be visible; Indoors: Image is set indoors, where the backdrop is in interior locations or permanent structures such as studios, rooms and in confined spaces (eg tunnels). Artificial lighting may also be present. Analyze the provided still image and reply with one of these exact labels: 'Graphics', 'Outdoor' or 'Indoor'."

        print(f"Prompt: {prompt}")

        confirm = input("Confirm? (y/n): \n")

    print(f'Prompt confirmed: {prompt}')

    column_name = "location_vlm_03" # Update column name
    model_name = 'qwen2.5-vl-7b-instruct'

    start_time = datetime.datetime.fromtimestamp(time.time())
    print(f'Starting annotation at {start_time}...')

    # Write log
    with open(log_file, "a") as log:
        log.write(f"\nVLM Annotation for column: {column_name} \n"
                  f"Model: {model_name} \n"
                  f"Prompt: {prompt} \n"
                  f"Start time: {start_time} \n\n"
                  f"--Log Started--\n")

    df_scene_spox = annotate_vlm(prompt=prompt,
                                 column_name=column_name,
                                 df_scene=df_scene_spox,
                                 frames_folder=frames_folder,
                                 model_name=model_name,
                                 )
    output_path = file_path

    df_scene_spox.to_csv(output_path, index=False)

    print()
    print(f'df_scene exported to {file_name}')
    print(f'Results written to {column_name}')
    end_time = datetime.datetime.fromtimestamp(time.time())
    print(f'Annotation ended at {end_time}...')
    elapsed_time = (end_time - start_time)

    # Add log information to log dataframe
    new_row = pd.DataFrame([{
        'column': column_name,
        'model': model_name,
        'prompt': prompt,
        'start_time': start_time,
        'end_time': end_time,
        'elapsed_time': elapsed_time,
    }])
    df_log = pd.concat([df_log, new_row], ignore_index=True)
    df_log.to_csv(log_data_path, index=False)

    with open(log_file, "a") as log:
        log.write(f'End time: {end_time}\n----------\n\n')

def get_qwen_prediction(image_path, prompt, model_name):
    '''
    :param image_path: (str) path to the image file.
    :param prompt: (str) prompt to be used for the model.
    :return: prediction
    '''
    # Check file validity
    # Valid image file extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')

    # Valid file extension check
    if not image_path.lower().endswith(valid_extensions):
        raise ValueError(f"Invalid file extension: {image_path}. Supported extensions are {valid_extensions}")

    # File existence check
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    # Check prompt validity
    if not isinstance(prompt, str):
        raise TypeError(f"The prompt must be a string, but got {type(prompt)}")
    if not prompt.strip():  # Empty or whitespace-only prompt
        raise ValueError("The prompt cannot be empty or whitespace. Please provide a valid prompt.")

    image_handle = lms.prepare_image(image_path)  # Convert to image_handle

    model = lms.llm(model_name)
    chat = lms.Chat()
    chat.add_user_message(prompt, images=[image_handle])
    try:
        prediction = model.respond(chat)
    except Exception as e:
        # Handle any exception that might occur during prompting
        raise RuntimeError(f"An error occurred during prompting: {str(e)}")

    # Regex to extract file name
    match = re.search(r'[^/]+$', image_path)
    if match:
        image_name = match.group()
        # print(f'Prediction of {image_name}: {prediction.content}')  # Output: 7290776842250833153_scene_1.jpeg
    else:
        print("Error in obtaining file name.")

    return prediction.content

def get_qwen_prediction_batch(df_scene, frames_folder, prompt, column_name, model_name):

    '''
    :param df_scene: dataframe object; Video scene data as generated by 4CAT
    :param frames_folder: file path str; Target folder path where the scene thumbnails are stored
    :param prompt: str; Prompt to be used by the Qwen model
    :param column_name: str;Name of the new column to be added to the dataframe
    :param model_name: str; Name of VLM to be used
    :return: df_scene: dataframe object; Updated dataframe with the Qwen predictions
    '''

    id_list = df_scene['url'].unique().tolist()
    total_scenes = len(df_scene)
    counter = 0
    # Loop through list of videos
    for video_id in id_list:
        # video_id example: '7295526580741229825'
        df_scenes = df_scene[df_scene['url'] == video_id]
        scene_list = df_scenes['id'] # Obtain list of scene ids in a video
        # Loop through scenes in specified video
        for scene_id in scene_list:
            # scene_id example '7295526580741229825.mp4_scene_1'
            cleaned_id = re.sub(r'\.mp4', '', scene_id)
            thumbnail_path = os.path.join(frames_folder, str(video_id), f'{cleaned_id}.jpeg')
            # print(f'Processing {cleaned_id}.jpeg...')

            prediction = get_qwen_prediction(thumbnail_path, prompt, model_name)
            counter += 1
            print(f'Prediction of {scene_id}: {prediction} ({counter}/{total_scenes})')
            if (counter) % 50 == 0:  # Log memory use for every 50 entries
                time_now = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
                with open(log_file, "a") as log:
                    log.write(f'{time_now} - {counter}/{total_scenes} scenes processed. \n')
            # Append to new column
            df_scene.loc[df_scene['id'] == scene_id, column_name] = prediction

    return df_scene

def annotate_vlm(prompt, column_name, df_scene, frames_folder, model_name='qwen2.5-vl-7b-instruct'):
    """
    Takes a specified vision-language model and prompt to analyze video thumbnails and updates the main scene metadata. Requires local developer server of LM Studio to be running with the desired model loaded. See https://lmstudio.ai/docs/app/api for more details.

    :param prompt: A string containing the text instruction for the vision-language model.
    :param column_name: The name of the column in the dataframe where predictions will be
        stored.
    :param model_name: Optional; name of the vision-language model to be used.
        Default is 'qwen2-vl-7b-instruct'.
    :param df_scene: A pandas dataframe containing the input data to be annotated.
    :param frames_folder: Directory path where the input frame files are located.
    :return: None
    """

    if model_name == '':
        model_name = 'qwen2-vl-7b-instruct'

    print(f'Prompting model \'{model_name}\' on LM Studio...')
    df_scene = get_qwen_prediction_batch(df_scene=df_scene,
                                         frames_folder=frames_folder,
                                         prompt=prompt,
                                         column_name=column_name,
                                         model_name=model_name
                                         ) # Reassign as needed; formatted it this way to avoid it being messed up by the order
    print()

    print('Done!')
    return df_scene


main()