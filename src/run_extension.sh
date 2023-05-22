ways=(2 5)
shots=(5)

# For each model run the python script for lstm leakage.
for way in ${ways[@]}; do
    for shot in ${shots[@]}; do

        echo "----------------------------"
        echo -e "Running the extension with $way ways and $shot shots"
        echo "----------------------------"
        filename="output_way${way}_shot${shot}.log"
        echo -e "Log file name is $filename"
        python3 -u extension.py --ways $way --shots $shot --file $filename
    done
done