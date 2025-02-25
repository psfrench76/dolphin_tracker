import os
import shutil
import re
import click


@click.command()
@click.argument('input_folder', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
@click.argument('start_frame', type=int)
@click.argument('end_frame', type=int)
def copy_frame_range(input_folder, output_folder, start_frame, end_frame):
    """
    Copy files from INPUT_FOLDER to OUTPUT_FOLDER whose frame numbers are between START_FRAME and END_FRAME.
    The range is not inclusive of the END_FRAME.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define the pattern to extract frame number
    pattern = re.compile(r'_(\d+)\.')

    # Initialize a counter for copied files
    copied_files_count = 0

    # Iterate over files in the input folder
    for filename in os.listdir(input_folder):
        match = pattern.search(filename)
        if match:
            frame_number = int(match.group(1))
            if start_frame <= frame_number < end_frame:
                src_path = os.path.join(input_folder, filename)
                dst_path = os.path.join(output_folder, filename)
                shutil.copy(src_path, dst_path)
                copied_files_count += 1

    # Print the final count of copied files
    click.echo(f"Total files copied: {copied_files_count}")


if __name__ == '__main__':
    copy_frame_range()
