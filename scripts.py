
# run the main.py script 50 times and this file will create an log file with the results
# the format of log file # output file
# output_file = open("../../TS-LoRa-sim/output.txt", "w")

import subprocess
import random

# Number of runs
num_runs = 50

for i in range(2, num_runs + 1):
    random_number = random.randint(10, 50)
    average_wake_up_time = random.randint(20, 40)
    simulation_time = random.randint(3600, 7200)
    # Construct the command to run the simulation, including the run number
    command = f'python3 main.py {random_number} 16 {average_wake_up_time} {simulation_time} {i}'
    subprocess.run(command, shell=True)
    print(f"Run {i} completed.")