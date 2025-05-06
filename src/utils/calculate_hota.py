from inc.reporting import HOTAContainer
import argparse
import pathlib

def main():
    parser = argparse.ArgumentParser(description="Calculate HOTA metrics.")
    parser.add_argument('--output_folder', '-o', type=pathlib.Path, required=True, help="Path to the output folder from the tracker.")
    args = parser.parse_args()
    output_folder = args.output_folder

    hc = HOTAContainer(output_folder)
    hc.run()

if __name__ == "__main__":
    main()

