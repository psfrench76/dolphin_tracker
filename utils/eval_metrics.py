import sys
import ast

def extract_metrics(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if 'results_dict:' in line:
                line = line.strip()
                # Extract the dictionary part from the line
                results_dict_str = line.split('results_dict: ')[1]
                # Convert the string representation of the dictionary to an actual dictionary
                results_dict = ast.literal_eval(results_dict_str)

                # Extract the required metrics
                precision = results_dict['metrics/precision(B)']
                recall = results_dict['metrics/recall(B)']
                map50 = results_dict['metrics/mAP50(B)']
                map50_95 = results_dict['metrics/mAP50-95(B)']

                # Print the metrics in a tab-separated format
                print(f"Precision\tRecall\tmAP50\tmAP50-95")
                print(f"{precision}\t{recall}\t{map50}\t{map50_95}")
                break

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python eval_metrics.py <path_to_txt_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    extract_metrics(file_path)