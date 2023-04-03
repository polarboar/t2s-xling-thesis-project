
. ./activate.sh
DATA_DIR=
LABEL_DIR=
FAIRSEQ_PATH=/disk/scratch1/ramons/myapps/t2s-xling/myapps/SpeechT5/SpeechT5/fairseq

python ${FAIRSEQ_PATH}/fairseq_cli/hydra_train.py \
  --config-dir /disk/scratch1/ramons/myapps/t2s-xling/myapps/SpeechT5/SpeechT5/speecht5/config/ \
  --config-name learning_rate
 # task.data=${DATA_DIR} task.label_dir=${LABEL_DIR} task.labels='["km"]' \
 # model.label_rate=50 common.user_dir=SpeechT5/Speech2C/speech2c 
