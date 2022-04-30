#!/bin/bash
cd "$(dirname "$0")"
commands="python3 train.py"
nohup bash -c "$commands" > process.log 2>&1 &
echo $! > process.pid
echo "started task, pid $(cat process.pid)"