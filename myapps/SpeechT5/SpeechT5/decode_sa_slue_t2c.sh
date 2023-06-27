
export CUDA_VISIBLE_DEVICES=0

. ./activate.sh


CHECKPOINT_PATH=/home/s2450029/repos/t2s-xling/models/speechT5/fairseq/speecht5_base_t2c/slue_sa/checkpoint_best.pt
DATA_ROOT=/home/s2450029/repos/t2s-xling/data_formatted/fairseq/text/tsv/sa_slue/
SUBSET=valid
USER_DIR=speecht5
RESULTS_PATH=/home/s2450029/repos/t2s-xling/models/speechT5/fairseq/slue_sa/result/t2c/
BPE_TOKENIZER=/home/s2450029/repos/t2s-xling/models/speechT5/fairseq/spm/spm_char.model

mkdir -p ${RESULTS_PATH}

python scripts/generate_class_from_text.py ${DATA_ROOT} \
  --gen-subset ${SUBSET} \
  --user-dir ${USER_DIR} \
  --log-format json \
  --task speecht5 \
  --t5-task t2c \
  --path ${CHECKPOINT_PATH} \
  --results-path ${RESULTS_PATH} \
  --batch-size 1 \
  --max-speech-positions 250000 \
  --sample-rate 16000 \
  --max-tokens 1400000 \
  --max-text-positions 600 \
  --bpe-tokenizer ${BPE_TOKENIZER}
