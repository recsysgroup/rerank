#!/usr/bin/env bash

#index=$RANDOM
index=27795
echo $index

RAW_TRAIN_PATH=~/Data/yahoo_c14_ltr/set1.train.txt
RAW_VALID_PATH=~/Data/yahoo_c14_ltr/set1.valid.txt
RAW_TEST_PATH=~/Data/yahoo_c14_ltr/set1.test.txt
TRAIN_PATH=/tmp/yahoo_c14_ltr/set1.train.txt
VALID_PATH=/tmp/yahoo_c14_ltr/set1.valid.txt
TEST_PATH=/tmp/yahoo_c14_ltr/set1.test.txt
MODEL_PATH=/tmp/model_${index}.txt

function middle_path() {
    type=$1
    echo '/tmp/yahoo_c14_ltr/set1.'${type}'.txt'
}

function output_path() {
    type=$1
    echo '/tmp/rerank_'${type}'_'${index}'.txt'
}

#mkdir /tmp/yahoo_c14_ltr
#echo 'cat '$RAW_TRAIN_PATH' | python2 data_complement.py > '$TRAIN_PATH
#cat $RAW_TRAIN_PATH | python2 data_complement.py > $TRAIN_PATH
#
#echo 'cat '$RAW_TEST_PATH' | python2 data_complement.py > '$TEST_PATH
#cat $RAW_TEST_PATH | python2 data_complement.py > $TEST_PATH
#
#echo 'cat '$RAW_VALID_PATH' | python2 data_complement.py > '$VALID_PATH
#cat $RAW_VALID_PATH | python2 data_complement.py > $VALID_PATH
#
#
#echo '###train'
#
#java -jar -Xms8194m -Xmx8194m RankLib-2.13.jar -train $TRAIN_PATH -test $TEST_PATH -validate $VALID_PATH -estop 1 -ranker 0 -metric2t MAP -save $MODEL_PATH

function add_noisy_data() {
  DATA_TYPE=$1
  echo $DATA_TYPE
  echo '###rank'

  DATA_PATH=`middle_path ${DATA_TYPE}`
  OUTPUT_PATH=`output_path ${DATA_TYPE}`
  TMP_PATH=/tmp/myScoreFile_${DATA_TYPE}_${index}.txt
  TMP_SCORE_PATH=/tmp/myScoreFile_score_${DATA_TYPE}_${index}.txt

  if [ -f $OUTPUT_PATH ]; then
    rm $OUTPUT_PATH
  fi

  if [ -f $TMP_PATH ]; then
    rm $TMP_PATH
  fi

  if [ -f $TMP_SCORE_PATH ]; then
    rm $TMP_SCORE_PATH
  fi


  java -jar -Xms8194m -Xmx8194m RankLib-2.13.jar -load $MODEL_PATH -rank $DATA_PATH -score $TMP_PATH

  echo '###data-prepare'

  cat $TMP_PATH | awk -F '\t' '{print $3}' > $TMP_SCORE_PATH

  paste -d ' ' $TMP_SCORE_PATH $DATA_PATH > $OUTPUT_PATH

  echo '###output: '$OUTPUT_PATH

  return 0;
}

add_noisy_data train
add_noisy_data valid
add_noisy_data test

