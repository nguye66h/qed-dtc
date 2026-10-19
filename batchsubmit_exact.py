import os
import subprocess

def read_inputs(inputs_filename):
    """Reads inputs.txt file and returns the job information."""
    jobs = []
    with open(inputs_filename, 'r') as file:
        lines = file.readlines()[1:]  # Skip header
        for line in lines:
            if line.strip():
                parts = line.split()
                job = {
                    'ep':         float(parts[0]),
                    'k':          float(parts[1]),
                    'type_of_coupling': parts[2],
                    'lambda':     float(parts[3]),
                    'per':        float(parts[4]),
                    'N':            int(parts[5]),
                    'ainit':            parts[6],
                    'sxinit':     float(parts[7]),
                    'syinit':     float(parts[8]),
                    'szinit':     float(parts[9]),
                    'method':           parts[10],
                    'integrator':       parts[11],
                    'tmax':       float(parts[12]),
                    'modulated':        parts[13],
                    'evolving':         parts[14],
                    'nphoton':      int(parts[15]),
                    'dt':         float(parts[16]),
                    'ntraj':        int(parts[17]),
                    'mem':          int(parts[18]),
                    'ntask':        int(parts[19])
                }
                jobs.append(job)

    for job in jobs:
        for i in range(job['ntask']):
            # Construct the command with the variables
            cmd = f"./jobsubmit_exact_cmd.sh {job['ep']} {job['k']} {job['type_of_coupling']} {job['lambda']} {job['per']} {job['N']} {job['ainit']} {job['sxinit']} {job['syinit']} {job['szinit']} {job['method']} {job['integrator']} {job['tmax']} {job['modulated']} {job['evolving']} {job['nphoton']} {job['dt']} {job['ntraj']} {job['mem']} {i}"
            # Execute the command without capturing the output
            subprocess.run(cmd, shell=True)

    return jobs

def main():
    inputs_filename = 'inputs.txt'

    # Read inputs
    jobs = read_inputs(inputs_filename)

if __name__ == "__main__":
    main()
