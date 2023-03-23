

. ./activate.sh


sst_folder=/disk/scratch1/ramons/data/t2s-xling/data/text/sst/data/sst/
sst_folder_dest=/disk/scratch2/ramons/data/t2s-xling/data_formated/fairseq/text/sst/

jq -e '.text' ${sst_folder}/train.jsonl | sed 's/\.\"//g' | sed 's/"//g' | awk '{print toupper($0)}' > ${sst_folder_dest}/train.txt
jq -e '.text' ${sst_folder}/test.jsonl | sed 's/\.\"//g' | sed 's/"//g'  | awk '{print toupper($0)}' > ${sst_folder_dest}/test.txt
jq -e '.text' ${sst_folder}/val.jsonl | sed 's/\.\"//g' | sed 's/"//g'   | awk '{print toupper($0)}' > ${sst_folder_dest}/val.txt

jq -e '.label' ${sst_folder}/train.jsonl | sed 's/\.\"//g' | sed 's/"//g' > ${sst_folder_dest}/train.label
jq -e '.label' ${sst_folder}/test.jsonl | sed 's/\.\"//g' | sed 's/"//g' > ${sst_folder_dest}/test.label
jq -e '.label' ${sst_folder}/val.jsonl | sed 's/\.\"//g' | sed 's/"//g' > ${sst_folder_dest}/val.label


spm_model=/disk/scratch2/ramons/data/t2s-xling/models/speechT5/fairseq/spm/spm_char.model
dict_file=/disk/scratch2/ramons/data/t2s-xling/models/speechT5/fairseq/spm/dict.txt

python decode_spm.py ${sst_folder_dest}/train.txt ${spm_model} ${sst_folder_dest}/train.spm.en
python decode_spm.py ${sst_folder_dest}/test.txt ${spm_model} ${sst_folder_dest}/test.spm.en
python decode_spm.py ${sst_folder_dest}/val.txt ${spm_model} ${sst_folder_dest}/val.spm.en


fairseq-preprocess \
--source-lang en \
--trainpref ${sst_folder_dest}/train.spm \
--validpref ${sst_folder_dest}/val.spm \
--testpref ${sst_folder_dest}/test.spm \
--destdir ${sst_folder_dest} \
--srcdict ${dict_file} \
--workers 20 \
--only-source \
--task speech_to_text \
