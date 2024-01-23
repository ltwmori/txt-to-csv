import csv
import os

# extracting header information from a log file
def extract_header(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    header_info = {}
    for line in lines:
        if line.startswith('Nodes:'):
            header_info['Nodes'] = line.split(':')[1].strip()
        elif line.startswith('Data size:'):
            header_info['Data Size'] = line.split(':')[1].replace('bytes', '').strip()
        elif line.startswith('Average wake up time of nodes'):
            header_info['Avg Wake Up Time'] = line.split(':')[1].strip().split(' ')[0].strip()  # extracting the number only
        elif line.startswith('Simulation time:'):
            header_info['Simulation Time'] = line.split(':')[1].strip().split(' ')[0].strip()  # extracting the number only
        elif line.strip() == '':  # empty line indicates the end of the header
            break

    return header_info

# Function to get the list of log files
def get_log_files():
    log_files = []
    i = 1
    while True:
        log_file_name = f'logs/log{i}.txt'
        if os.path.exists(log_file_name):
            log_files.append(log_file_name)
            i += 1
        else:
            break
    return log_files

# Get the list of log files
log_files = get_log_files()

with open('headers.csv', 'w', newline='') as csvfile:
    fieldnames = ['Log File ID', 'Nodes', 'Data Size', 'Avg Wake Up Time', 'Simulation Time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for log_file in log_files:
        header_info = extract_header(log_file)
        if header_info:
            log_file_id = log_file.replace('.txt', '').replace('logs/', '')
            header_info['Log File ID'] = log_file_id
            writer.writerow(header_info)

print("Headers.csv file has been created.")
