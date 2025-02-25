import subprocess
import click
import os
from PIL import Image


@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
def main(input_video, output_folder):
    extract_frames(input_video, output_folder)


def extract_frames(input_video, output_folder):
    """
    Extract frames from INPUT_VIDEO and save them as jpg files in OUTPUT_FOLDER.
    The frames will be named [original_mp4_name]_[frame_number].jpg with zero-padded frame numbers.
    """

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the video name without extension
    video_name = os.path.splitext(os.path.basename(input_video))[0]

    # Use FFmpeg with GPU acceleration to extract frames
    ffmpeg_command = [
        'ffmpeg',
        # '-hwaccel', 'cuda',  # Use CUDA for hardware acceleration if available
        '-i', input_video,
        f'{output_folder}/{video_name}_%06d.jpg'
    ]

    subprocess.run(ffmpeg_command)

    click.echo(f"Frames extracted from {input_video} into {output_folder}.")

    # Get the height of one of the extracted frames
    first_frame_path = os.path.join(output_folder, f'{video_name}_000001.jpg')
    with Image.open(first_frame_path) as img:
        image_height = img.height

    return image_height

if __name__ == '__main__':
    main()
