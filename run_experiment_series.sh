#! /bin/sh

for pars in $(seq 1 1)
do
    python3 run_experiment.py $pars
done
