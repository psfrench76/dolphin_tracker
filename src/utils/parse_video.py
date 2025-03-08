import subprocess
import click
import os
from PIL import Image
from inc.video_processing import extract_frames


@click.command()
@click.argument('input_video', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
def main(input_video, output_folder):
    extract_frames(input_video, output_folder)


if __name__ == '__main__':
    main()
