import os
import click

@click.command()
@click.argument('image_folder', type=click.Path(exists=True))
@click.argument('label_folder', type=click.Path())
def create_blank_labels(image_folder, label_folder):
    """
    Create empty .txt files in LABEL_FOLDER for each .jpg in IMAGE_FOLDER if they don't already exist.
    """
    # Ensure the label folder exists
    os.makedirs(label_folder, exist_ok=True)

    # Initialize counters
    created_count = 0
    skipped_count = 0

    # Iterate over .jpg files in the image folder
    for filename in os.listdir(image_folder):
        if filename.endswith('.jpg'):
            base_name = os.path.splitext(filename)[0]
            label_file = os.path.join(label_folder, f"{base_name}.txt")
            if not os.path.exists(label_file):
                open(label_file, 'w').close()
                click.echo(f"Created: {label_file}")
                created_count += 1
            else:
                skipped_count += 1

    # Print the final counts
    click.echo(f"Total files created: {created_count}")
    click.echo(f"Total files skipped: {skipped_count}")

if __name__ == '__main__':
    create_blank_labels()