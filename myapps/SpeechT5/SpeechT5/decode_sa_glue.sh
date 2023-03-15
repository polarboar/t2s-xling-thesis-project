
export CUDA_VISIBLE_DEVICES=0

. ./activate.sh


SPEECHT5_CODE_DIR=
CHECKPOINT_PATH=/disk/scratch1/ramons/data/t2s-xling/models/speechT5/fairseq/slue_sa/checkpoint_best.pt
DATA_ROOT=/disk/scratch1/ramons/data/t2s-xling/data_formated/fairseq/text/tsv/sa_glue/
SUBSET=valid
BPE_TOKENIZER=
LABEL_DIR=
USER_DIR=speecht5
RESULTS_PATH=/disk/scratch1/ramons/data/t2s-xling/models/speechT5/fairseq/slue_sa/result

mkdir -p ${RESULTS_PATH}


python3 ${SPEECHT5_CODE_DIR}/SpeechT5/scripts/generate_speech.py ${DATA_ROOT} \
  --gen-subset ${SUBSET} \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  --user-dir ${USER_DIR} \
  --task speecht5 \
  --t5-task t2s \
  --path ${CHECKPOINT_PATH} \
  --hubert-label-dir ${LABEL_DIR} \
  --batch-size 1 \
  --results-path ${RESULTS_PATH} \
  --sample-rate 16000

