ENCODEC_SET=encodec320x_ratios8542
DATASET=libri_v1
python train_multi_gpu.py
                    hydra.run.dir=${WORK}/hydra_outputs/${ENCODEC_SET}_${DATASET}