# !/usr/bin/env bash
export OSS_HOST=cn-hangzhou.oss-internal.aliyun-inc.com
export ROLE_ARN=acs:ram::1384235136031533:role/jiusheng
export SIMULATOR_CHECKPOINT_PATH=/tmp/yahoo_gru_v4/model.ckpt-238

export TRAIN_TABLE=~/Data/yahoo_c14_ltr/yahoo_seq_train.txt
export TEST_TABLE=~/Data/yahoo_c14_ltr/yahoo_seq_test.txt
export CHECKPOINT_PATH=/tmp/yahoo_gru_rl_v10

#tar -czf /tmp/rl_rerank.tar.gz .
#
#odpscmd -e 'use etao_backend;
#pai -name tensorflow140
#-Dscript="file:///tmp/rl_rerank.tar.gz"
#-DentryFile="src/main.py"
#-Dbuckets="oss://jiusheng-tmp/?host='${OSS_HOST}'&role_arn='${ROLE_ARN}'"
#-Dtables="'${TRAIN_TABLE}','${EVAL_TABLE}'"
#-Dcluster="{\"ps\":{\"count\":1},\"worker\":{\"count\":32}}"
#-DuserDefinedParameters="--num_train_steps=200000 --task_type=train --env=server --config=src/scripts/yahoo_gru_rl_20_config.json --train_table='${TRAIN_TABLE}' --eval_table='${EVAL_TABLE}' --checkpoint_path='${CHECKPOINT_CONT_DIR}' --simulator_checkpoint_path='${SIMULATOR_CHECKPOINT_PATH}'"
#'
# pai -name tensorboard -DsummaryDir='oss://yiyang-yy/rl_rerank/checkpoint/onion_single/?host=oss-cn-zhangjiakou-internal.aliyuncs.com&role_arn=acs:ram::1590763244468021:role/yiyang'

 python2 src/main.py  --num_train_steps=200000 --task_type=train --env=server --config=src/scripts/yahoo_gru_rl_20_config.json --train_table=${TRAIN_TABLE} --eval_table=${TEST_TABLE} --checkpoint_path=${CHECKPOINT_PATH} --simulator_checkpoint_path=${SIMULATOR_CHECKPOINT_PATH}


# eval 2019110322 auc:0.697974443436