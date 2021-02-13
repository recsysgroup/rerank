#!/usr/bin/env bash

#index=$RANDOM
index=18713
echo $index

TRAIN_PATH=/tmp/yahoo_click_train.txt
VALID_PATH=/tmp/yahoo_click_valid.txt
TEST_PATH=/tmp/yahoo_click_test.txt
MODEL_PATH=/tmp/model_${index}.txt
RERANK_PATH=/tmp/rerank_${index}.txt


echo '###train'

java -jar RankLib-2.13.jar -train $TRAIN_PATH -test $TEST_PATH -validate $VALID_PATH -ranker 7 -epoch 100 -metric2t MAP -save $MODEL_PATH

echo '###rank'

java -jar RankLib-2.13.jar -load $MODEL_PATH -rank $TEST_PATH -score /tmp/myScoreFile_${index}.txt

echo '###data-prepare'

cat /tmp/myScoreFile_${index}.txt | awk -F '\t' '{print $3}' > /tmp/myScoreFile_score_${index}.txt

paste -d ' ' /tmp/myScoreFile_score_${index}.txt $TEST_PATH > $RERANK_PATH

#rm /tmp/myScoreFile_${index}.txt
#rm /tmp/myScoreFile_score_${index}.txt

echo '###evaluate'

python evaluator.py $RERANK_PATH map
