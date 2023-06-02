
PATH_VOXCELEB=/home/s2450029/repos/slue-toolkit/data/slue-voxceleb
PATH_DESTINATION=/home/s2450029/repos/t2s-xling/data_formatted/fairseq/speech/tsv/sa_slue

#echo $PATH_VOXCELEB/fine-tune_raw
#echo "$PATH_DESTINATION/train.tsv"



echo "$PATH_VOXCELEB/fine-tune_raw" > $PATH_DESTINATION/train.tsv
cat $PATH_VOXCELEB/slue-voxceleb_fine-tune.tsv | awk -v FS='\t' '{if($5!="<mixed>")print $1".flac\t"($NF-$(NF-1))*16000"\t"$(NF-2)}' | tail  -n +2 >> $PATH_DESTINATION/train.tsv

echo "$PATH_VOXCELEB/dev_raw" > $PATH_DESTINATION/valid.tsv
cat $PATH_VOXCELEB/slue-voxceleb_dev.tsv | awk -v FS='\t' '{if($5!="<mixed>" && $5!="Disagreement")print $1".flac\t"($NF-$(NF-1))*16000"\t"$(NF-2)}' |  tail  -n +2 >> $PATH_DESTINATION/valid.tsv

cat $PATH_VOXCELEB/slue-voxceleb_dev.tsv $PATH_VOXCELEB/slue-voxceleb_fine-tune.tsv | awk -v FS='\t' '{if($5!="<mixed>" && $5!="Disagreement" && $5!="sentiment")print $(NF-2)}' | sort | uniq | awk '{print $1" "1}' > $PATH_DESTINATION/dict.txt

