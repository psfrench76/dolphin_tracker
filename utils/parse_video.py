import click
import os
import cv2

@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
def extract_frames(input_video, output_folder):
    """
    Extract frames from INPUT_VIDEO and save them as jpg files in OUTPUT_FOLDER.
    The frames will be named [original_mp4_name]_[frame_number].jpg.
    """

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the video name without extension
    video_name = os.path.splitext(os.path.basename(input_video))[0]

    # Open the video file
    cap = cv2.VideoCapture(input_video)
    frame_number = 0
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Construct the filename
        frame_filename = f"{video_name}_{frame_number:06}.jpg"
        frame_filepath = os.path.join(output_folder, frame_filename)

        # Save the frame as a jpg file
        cv2.imwrite(frame_filepath, frame)

        frame_number += 1

        # Show progress
        print(f"Processing frame {frame_number}/{total_frames}", end='\r')

    cap.release()
    print(f"\nExtracted {frame_number} frames from {input_video}.")

if __name__ == '__main__':
    extract_frames()