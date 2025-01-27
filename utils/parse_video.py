import subprocess
import click
import os

@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
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
        #'-hwaccel', 'cuda',  # Use CUDA for hardware acceleration if available
        '-i', input_video,
        f'{output_folder}/{video_name}_%06d.jpg'
    ]

    subprocess.run(ffmpeg_command)

    click.echo(f"Frames extracted from {input_video} into {output_folder}.")

if __name__ == '__main__':
    extract_frames()