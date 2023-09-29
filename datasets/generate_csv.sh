#!/bin/bash
# Create a csv description files for train and test
input_file='data/LibriSpeech/test-clean'
output_file='librispeech_dataset.csv'
python generate_train_file.py -i $input_file -o $output_file -s
echo "Generated csv file"