import csv
import os 

def extract_node_initialization(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    start_capture = False
    node_init_data = []

    for line in lines:
        if line.strip() == 'Node initialization:':
            start_capture = True
            continue
        elif line.strip() == 'Simulation start':
            break
        elif start_capture:
            parts = line.split()
            node_id = parts[1].replace(':', '')
            x_coord = parts[3]
            y_coord = parts[5]
            distance = parts[7]
            sf = parts[9]
            node_init_data.append([node_id, x_coord, y_coord, distance, sf])

    return node_init_data


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

log_files = get_log_files()  # Assuming this function is defined to get log files

# Writing Node Initialization data to CSV
with open('node_initialization.csv', 'w', newline='') as csvfile:
    fieldnames = ['Log File ID', 'Node ID', 'X Coord', 'Y Coord', 'Distance', 'SF']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)

    for log_file in log_files:
        node_init_data = extract_node_initialization(log_file)
        for row in node_init_data:
            writer.writerow([log_file.replace('.txt', '').replace('logs/', '')] + row)
print("node_initialization.csv file has been created.")
