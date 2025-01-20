#!/bin/bash

submit_server="hpc"
sbatch_file="queue-dolphin-server.sbatch"
tunnel_file="tunnel-dolphin-server.sh"
server_error_file="jupyter/dolphin-server.err"
utils_dir="/nfs/stak/users/frenchp/AI506/dolphin_tracker/dolphin_tracker/utils/hpc"

submit_command="ssh $submit_server"
echo "submitting slurm job..."
# Start Slurm Job
job_info=$($submit_command "/apps/slurm/bin/sbatch $utils_dir/$sbatch_file")
job_id=$(echo $job_info | awk '{print $4}')

echo "Waiting for slurm job to start..."
while true; do
  job_status=$($submit_command "squeue -h -j $job_id -o %T" 2>/dev/null)
  if [ "$job_status" == "RUNNING" ]; then
    break
  fi
  sleep 5
done
echo "Slurm job is now running."

# Get Slurm job server and open ssh tunnels locally
allocated_server=$($submit_command "squeue -j $job_id --format=%N --noheader")
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1090
# Open ssh tunnels for ports 8080 and 6999 for jupyter and tensorboard servers
source "$script_dir/$tunnel_file" "$allocated_server"

# Start Jupyter server
echo "Waiting for Jupyter notebook to start..."
while true; do
# Get token from Z-drive log on submit server
  token=$($submit_command "grep http://localhost:8080/tree?token= -m 1 ~/logs/$server_error_file | sed -n -e s/.*=//p" 2>/dev/null)
  if [ -n "$token" ]; then
    break
  fi
  sleep 5
done
echo "Jupyter notebook is now running."
open "http://localhost:8080/?token=$token"
open "http://localhost:6999/"

