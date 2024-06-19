import json
import argparse
import os
import csv


def load_json_files(directory):
    json_data = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('json'):
                with open(directory+file, 'r') as f:
                    data = json.load(f)
                    json_data.setdefault('traceEvents',[]).append(data['traceEvents'])

    return json_data

def parse(json_file_path, function_name, output_file_name):

    data = load_json_files(json_file_path)

    kernels=[]
    for entries in data["traceEvents"]:
        for entry in entries:
            print(entry)
            if  'name'in entry and ((entry['name'] == 'hipExtLaunchKernel' and function_name in entry['args']['args']) or function_name in entry['name']):
                kernels.append(entry)
                print(entry)

    sorted_kernels = sorted(kernels, key=lambda x: (x['args']['BeginNs'], x['args']['pid']))

    csv_file_name = output_file_name + '.csv'
    json_file_out = output_file_name + '.json'

    json_data_out = {}
    json_data_out.setdefault('traceEvents',[]).append({})

    with open(csv_file_name, 'w', newline='') as csvfile:
        fieldnames = ['pid', 'BeginNs', 'dur', 'ts']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        i = 1
        for entry in sorted_kernels:
            record = {'pid': entry['args']['pid'], 'BeginNs': entry['args']['BeginNs'], 'dur': entry['dur'],
                     'ts': entry['ts']}
            json_data_out.setdefault('traceEvents',[]).append(entry)

            writer.writerow(record)
            if (i) % 8 == 0:
                csvfile.write('\n')
                i = 0
                i = i + 1

        with open(json_file_out, 'w') as jsonfileout:
            json.dump(json_data_out, jsonfileout, indent=4)

    print(f"Data successfully written to {csv_file_name} and {json_file_out}.")

def main():
    parser = argparse.ArgumentParser(description='Json file and the function to parse.')

    #parser.add_argument('json_files_path', type=argparse.FileType('r'), help='JSON file to load!')
    parser.add_argument('json_file_path', metavar='file_path', type=str,  help='Path to the JSON file to process')
    parser.add_argument('function_name', type=str, help='Kernel Function Name, e.g., gatherTopK, ncclDevKernel_Generic, mscclKernel')
    parser.add_argument('output_file_name', type=str, help='Output File Name')

    args = parser.parse_args()
    parse(args.json_file_path, args.function_name, args.output_file_name)

if __name__ == '__main__':
    main()
