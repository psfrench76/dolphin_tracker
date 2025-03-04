import os
import click
import re


@click.command()
@click.argument('label_folder', type=click.Path(exists=True))
@click.argument('start_frame', type=int)
@click.argument('end_frame', type=int)
def remove_blank_labels(label_folder, start_frame, end_frame):
    """
    Remove empty .txt files in LABEL_FOLDER within the frame range START_FRAME to END_FRAME (inclusive).
    """
    # Initialize counters
    removed_count = 0
    skipped_count = 0

    # Define the pattern to extract frame number
    pattern = re.compile(r'_(\d+)\.txt')

    # Iterate over .txt files in the label folder
    for filename in os.listdir(label_folder):
        if filename.endswith('.txt'):
            match = pattern.search(filename)
            if match:
                frame_number = int(match.group(1))
                if start_frame <= frame_number <= end_frame:
                    file_path = os.path.join(label_folder, filename)
                    if os.path.getsize(file_path) == 0:
                        os.remove(file_path)
                        click.echo(f"Removed: {file_path}")
                        removed_count += 1
                    else:
                        skipped_count += 1

    # Print the final counts
    click.echo(f"Total files removed: {removed_count}")
    click.echo(f"Total files skipped: {skipped_count}")


if __name__ == '__main__':
    remove_blank_labels()
