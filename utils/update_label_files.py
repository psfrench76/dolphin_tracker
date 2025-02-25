import os
import shutil
import click


def index_files(directory):
    file_index = {}
    for root, _, files in os.walk(directory):
        for file in files:
            file_index[file] = os.path.join(root, file)
    return file_index


def update_label_files(original_label_directory, new_label_directory):
    # Index the files in the original_label_directory
    original_files = index_files(original_label_directory)

    total_files = len(os.listdir(new_label_directory))
    copied_files = 0
    missing_files = 0
    missing_files_list = []

    # Iterate over all files in the new_label_directory
    for i, new_label_file in enumerate(os.listdir(new_label_directory), start=1):
        new_label_file_path = os.path.join(new_label_directory, new_label_file)

        # Check if the file exists in the original_files index
        if new_label_file in original_files:
            original_label_file_path = original_files[new_label_file]

            # Replace the file in new_label_directory with the file from original_label_directory
            shutil.copyfile(original_label_file_path, new_label_file_path)
            copied_files += 1
        else:
            missing_files += 1
            missing_files_list.append(new_label_file)

        # Print progress
        print(f"Processed {i}/{total_files} files\r", end='')

    # Print summary
    print(f"\nTotal files processed: {total_files}")
    print(f"Files copied: {copied_files}")
    print(f"Files not found in original directory: {missing_files}")

    if missing_files_list:
        print("Files not found:")
        for missing_file in sorted(missing_files_list):
            print(missing_file)


@click.command()
@click.argument('original_label_directory', type=click.Path(exists=True))
@click.argument('new_label_directory', type=click.Path(exists=True))
def main(original_label_directory, new_label_directory):
    update_label_files(original_label_directory, new_label_directory)


if __name__ == "__main__":
    main()
