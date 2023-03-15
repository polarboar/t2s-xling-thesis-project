
export CUDA_VISIBLE_DEVICES=0

. ./activate.sh


CHECKPOINT_PATH=/disk/scratch1/ramons/data/t2s-xling/models/speechT5/fairseq/slue_sa/checkpoint_best.pt
DATA_ROOT=/disk/scratch1/ramons/data/t2s-xling/data_formated/fairseq/speech/tsv/sa_slue/
SUBSET=valid
USER_DIR=speecht5
RESULTS_PATH=/disk/scratch1/ramons/data/t2s-xling/models/speechT5/fairseq/slue_sa/result

mkdir -p ${RESULTS_PATH}

python scripts/generate_class.py ${DATA_ROOT} \
  --gen-subset ${SUBSET} \
  --user-dir ${USER_DIR} \
  --log-format json \
  --task speecht5 \
  --t5-task s2c \
  --path ${CHECKPOINT_PATH} \
  --results-path ${RESULTS_PATH} \
  --batch-size 1 \
  --max-speech-positions 8000 \
  --sample-rate 16000
