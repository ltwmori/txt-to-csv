import csv
import os 

def extract_simulation_start(log_file_path):
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    start_capture = False
    simulation_data = []

    for line in lines:
        if line.strip() == 'Simulation start':
            start_capture = True
            continue
        elif line.strip() == 'Simulation finished':
            break
        elif start_capture:
            parts = line.split()
            timestamp = parts[0]
            node_action = ' '.join(parts[2:])
            simulation_data.append([timestamp, node_action])

    return simulation_data


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
# Writing Simulation Start data to CSV
with open('simulation_start.csv', 'w', newline='') as csvfile:
    fieldnames = ['Log File ID', 'Timestamp', 'Node Action']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)

    for log_file in log_files:
        simulation_data = extract_simulation_start(log_file)
        for row in simulation_data:
            writer.writerow([log_file.replace('.txt', '').replace('logs/', '')] + row)
print("simulation_start.csv file has been created.")
