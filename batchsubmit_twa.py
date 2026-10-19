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
                    'w':         float(parts[0]),
                    'w0':        float(parts[1]),
                    'k':          float(parts[2]),
                    'type_of_coupling': parts[3],
                    'lambda':     float(parts[4]),
                    'per':        float(parts[5]),
                    'N':            int(parts[6]),
                    'ainit':            parts[7],
                    'sxinit':     float(parts[8]),
                    'syinit':     float(parts[9]),
                    'szinit':     float(parts[10]),
                    'tmax':       float(parts[11]),
                    'modulated':        parts[12],
                    'dt':         float(parts[13]),
                    'ntraj':        int(parts[14]),
                    'ntask':      int(parts[15]),  # Assuming ntraj and ntask are the same
                    'mem':          int(parts[16])
                }
                jobs.append(job)

    for job in jobs:
        for nt in range(job['ntask']):
            # Construct the command with the variables
            cmd = f"./jobsubmit_twa_cmd.sh {job['w']} {job['w0']} {job['k']} {job['type_of_coupling']} {job['lambda']} {job['per']} {job['N']} {job['ainit']} {job['sxinit']} {job['syinit']} {job['szinit']} {job['tmax']} {job['modulated']} {job['dt']} {job['ntraj']} {nt} {job['mem']}"
            # Execute the command without capturing the output
            subprocess.run(cmd, shell=True)

    return jobs

def main():
    inputs_filename = 'inputs_twa.txt'

    # Read inputs
    jobs = read_inputs(inputs_filename)

if __name__ == "__main__":
    main()
