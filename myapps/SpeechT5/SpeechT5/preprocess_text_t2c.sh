

. ./activate.sh

sst_folder=/home/s2450029/repos/slue-toolkit/data/slue-voxceleb
sst_folder_dest=/home/s2450029/repos/t2s-xling/data_formatted/fairseq/text/tsv/sa_slue

cat $sst_folder/slue-voxceleb_fine-tune.tsv | awk -v FS='\t' '{if (NR>1 && $5!="<mixed>") print $2}' > $sst_folder_dest/train.tsv
cat $sst_folder/slue-voxceleb_dev.tsv | awk -v FS='\t' '{if (NR>1 && $5!="<mixed>" && $5!="Disagreement") print $2}' > $sst_folder_dest/valid.tsv
#cat $sst_folder/slue-voxceleb_dev.tsv | awk -v FS='\t' '{if (NR>1) print $2}' > $sst_folder_dest/valid.tsv

cat $sst_folder/slue-voxceleb_fine-tune.tsv | awk -v FS='\t' '{if (NR>1 && $5!="<mixed>") print $5}' > $sst_folder_dest/train.labels
cat $sst_folder/slue-voxceleb_dev.tsv | awk -v FS='\t' '{if (NR>1 && $5!="<mixed>" && $5!="Disagreement") print $5}' > $sst_folder_dest/valid.labels


#jq -e '.text' ${sst_folder}/train.jsonl | sed 's/\.\"//g' | sed 's/"//g' | awk '{print toupper($0)}' > ${sst_folder_dest}/train.txt
#jq -e '.text' ${sst_folder}/test.jsonl | sed 's/\.\"//g' | sed 's/"//g'  | awk '{print toupper($0)}' > ${sst_folder_dest}/test.txt
#jq -e '.text' ${sst_folder}/val.jsonl | sed 's/\.\"//g' | sed 's/"//g'   | awk '{print toupper($0)}' > ${sst_folder_dest}/val.txt

#jq -e '.label' ${sst_folder}/train.jsonl | sed 's/\.\"//g' | sed 's/"//g' > ${sst_folder_dest}/train.label
#jq -e '.label' ${sst_folder}/test.jsonl | sed 's/\.\"//g' | sed 's/"//g' > ${sst_folder_dest}/test.label
#jq -e '.label' ${sst_folder}/val.jsonl | sed 's/\.\"//g' | sed 's/"//g' > ${sst_folder_dest}/val.label

spm_model=/home/s2450029/repos/t2s-xling/models/speechT5/fairseq/spm/spm_char.model
dict_file=/home/s2450029/repos/t2s-xling/models/speechT5/fairseq/spm/dict.txt

python decode_spm.py ${sst_folder_dest}/train.tsv ${spm_model} ${sst_folder_dest}/train.spm.en
python decode_spm.py ${sst_folder_dest}/valid.tsv ${spm_model} ${sst_folder_dest}/valid.spm.en
#python decode_spm.py ${sst_folder_dest}/val.txt ${spm_model} ${sst_folder_dest}/val.spm.en


fairseq-preprocess \
--source-lang en \
--trainpref ${sst_folder_dest}/train.spm \
--validpref ${sst_folder_dest}/valid.spm \
--destdir ${sst_folder_dest} \
--srcdict ${dict_file} \
--workers 20 \
--only-source \
--task speech_to_text \

mv ${sst_folder_dest}/dict.en.txt ${sst_folder_dest}/dict.txt
