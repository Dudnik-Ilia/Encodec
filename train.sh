ENCODEC_SET=encodec320x_ratios8542
DATASET=libri_v1
python train_multi_gpu.py
                    datasets.fixed_length=500 \
                    hydra.run.dir=/hydra_outputs/${ENCODEC_SET}_${DATASET}