#!/usr/bin/env bash


#odpscmd -e "tunnel upload yahoo_dataset/set1.train.txt dataset_yahoo_l2tc_train_raw"
#odpscmd -e "tunnel upload yahoo_dataset/set1.valid.txt dataset_yahoo_l2tc_valid_raw"
#odpscmd -e "tunnel upload yahoo_dataset/set1.test.txt dataset_yahoo_l2tc_test_raw"

#odpscmd -e "tunnel download tmp_yahoo_data_train_dump tmp_yahoo_data_train_dump.txt"
#odpscmd -e "tunnel download tmp_yahoo_data_valid_dump tmp_yahoo_data_valid_dump.txt"
#odpscmd -e "tunnel download tmp_yahoo_data_test_dump tmp_yahoo_data_test_dump.txt"

#index=$RANDOM
index=4962
echo $index

#echo '###train'
#
#java -Xms2g -Xmx8g -jar RankLib-2.13.jar -train yahoo_dataset/yahoo_rating_train.txt -test yahoo_dataset/yahoo_rating_test.txt -validate yahoo_dataset/yahoo_rating_valid.txt -ranker 0 -metric2t MAP -save /tmp/model_${index}.txt

echo '###rank'

#java -Xms2g -Xmx8g -jar RankLib-2.13.jar -load /tmp/model_${index}.txt  -rank yahoo_dataset/yahoo_rating_train.txt -score /tmp/myScoreFile_${index}_train.txt

cat /tmp/myScoreFile_${index}_train.txt | awk -F '\t' '{print $3}' > /tmp/myScoreFile_score_${index}_train.txt

paste -d ' ' /tmp/myScoreFile_score_${index}_train.txt yahoo_dataset/yahoo_rating_train.txt > /tmp/rerank_${index}_train.txt

#java -Xms2g -Xmx8g -jar RankLib-2.13.jar -load /tmp/model_${index}.txt  -rank yahoo_dataset/yahoo_rating_test.txt -score /tmp/myScoreFile_${index}_test.txt

cat /tmp/myScoreFile_${index}_valid.txt | awk -F '\t' '{print $3}' > /tmp/myScoreFile_score_${index}_valid.txt

paste -d ' ' /tmp/myScoreFile_score_${index}_valid.txt yahoo_dataset/yahoo_rating_valid.txt > /tmp/rerank_${index}_valid.txt

#java -Xms2g -Xmx8g -jar RankLib-2.13.jar -load /tmp/model_${index}.txt  -rank yahoo_dataset/yahoo_rating_test.txt -score /tmp/myScoreFile_${index}_test.txt

cat /tmp/myScoreFile_${index}_test.txt | awk -F '\t' '{print $3}' > /tmp/myScoreFile_score_${index}_test.txt

paste -d ' ' /tmp/myScoreFile_score_${index}_test.txt yahoo_dataset/yahoo_rating_test.txt > /tmp/rerank_${index}_test.txt
