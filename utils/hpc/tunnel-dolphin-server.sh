#!/bin/bash

remote_alias="hpc"

if [ -z "$1" ]; then
    echo "Please specify a server name"
    exit 1
fi

allocated_server=$1
servname="$allocated_server"

formatted_server="frenchp@$servname.hpc.engr.oregonstate.edu"

ssh -o StrictHostKeyChecking=no -N -L 8080:localhost:8080 -J $remote_alias $formatted_server &
ssh -o StrictHostKeyChecking=no -N -L 6999:localhost:6999 -J $remote_alias $formatted_server &
