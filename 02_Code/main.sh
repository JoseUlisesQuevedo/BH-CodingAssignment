#!/bin/bash

values=(100 500 1000 2000)

for value in "${values[@]}"
do
    echo "Running for $value"
    python 02_Code/data_prep.py $value
    python 02_Code/modeling.py $value

done

python 02_Code/generate_report_figures.py 



