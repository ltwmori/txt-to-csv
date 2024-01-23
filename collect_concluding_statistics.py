import csv
import os

# Function to extract concluding statistics from a log file
def extract_concluding_statistics(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    start_capture = False
    stats = {}

    for line in lines:
        if line.startswith('Simulation finished'):
            start_capture = True
            continue

        if start_capture and ':' in line:
            key, value = line.split(':', 1)
            key_formatted = key.strip()  # Use the exact key as in the log file
            value_formatted = value.strip().split(' ')[0]  # Extract only the number
            stats[key_formatted] = value_formatted

    return stats

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

# Define the statistics fields
stat_fields = [
    'Join Collisions', 'Data collisions', 'Lost packets (due to path loss)', 'Transmitted data packets',
    'Transmitted SACK packets', 'Missed SACK packets', 'Transmitted join request packets',
    'Transmitted join accept packets', 'Join Retransmissions', 'Data Retransmissions',
    'Join request packets dropped by gateway', 'Average join time', 'Average energy consumption (Rx)',
    'Average energy consumption (Tx)', 'Average energy consumption per node', 'PRR',
    'Number of nodes failed to connect to the network'
]

# CSV file creation
with open('concluding_statistics.csv', 'w', newline='') as csvfile:
    fieldnames = ['Log File ID'] + stat_fields
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for log_file in log_files:
        stats = extract_concluding_statistics(log_file)
        log_file_id = log_file.replace('.txt', '').replace('logs/', '')
        stats_row = {'Log File ID': log_file_id}
        stats_row.update({field: stats.get(field, 'N/A') for field in stat_fields})
        writer.writerow(stats_row)

print("concluding_statistics.csv file has been created.")
