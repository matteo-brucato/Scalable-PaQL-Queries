#!/usr/bin/env bash


python_script=`echo "$1" | tr "/" "."`
python_script=${python_script%".py"}

if [ ${python_script} ]; then
    shift
    python -m ${python_script} "$@"
else
    # If no Python script is given, just run a Python interpreter with all
    # PB libraries loaded up
    export PYTHONSTARTUP="src/utils/pb_startup_python_interpreter.py"
    ipython
fi
