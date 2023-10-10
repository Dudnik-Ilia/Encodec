#!/bin/bash
# Create a csv description files for train and test
input_file='$TMPDIR/$SLURM_JOBID/datasets/data/LibriSpeech/train-clean-100'
output_file='$TMPDIR/$SLURM_JOBID/datasets/librispeech_dataset_train.csv'
python generate_desc_file.py -i $input_file -o $output_file
echo "Generated TRAIN csv file"

input_file='$TMPDIR/$SLURM_JOBID/datasets/data/LibriSpeech/test-clean'
output_file='$TMPDIR/$SLURM_JOBID/datasets/librispeech_dataset_test.csv'
python generate_desc_file.py -i $input_file -o $output_file
echo "Generated TEST csv file"