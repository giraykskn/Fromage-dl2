echo "----------------------------"
echo -e "Running the extension with 2 ways and 1 shots and 0 repeats"
echo "----------------------------"
filename="output_way${2}_shot${1}_repeat${0}.log"
echo -e "Log file name is $filename"
python3 -u extension.py --ways 2 --shots 1 --repeats 1 --file $filename