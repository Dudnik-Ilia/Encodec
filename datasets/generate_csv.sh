#!/bin/bash
# Create a csv description files for train and test
TRAIN_WITHOUT_EXT="${TRAIN_FILE%.*}"
TEST_WITHOUT_EXT="${TEST_FILE%.*}"

input_file='$TMPDIR/$SLURM_JOBID/datasets/data/LibriSpeech/$TRAIN_WITHOUT_EXT'
output_file='$TMPDIR/$SLURM_JOBID/datasets/librispeech_dataset_train.csv'
python generate_desc_file.py -i $input_file -o $output_file
echo "Generated TRAIN csv file"

input_file='$TMPDIR/$SLURM_JOBID/datasets/data/LibriSpeech/$TEST_WITHOUT_EXT'
output_file='$TMPDIR/$SLURM_JOBID/datasets/librispeech_dataset_test.csv'
python generate_desc_file.py -i $input_file -o $output_file
echo "Generated TEST csv file"