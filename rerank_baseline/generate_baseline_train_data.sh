#!/usr/bin/env bash

cat yahoo_dataset/yahoo_nosiy_train.txt | python generate_scan_seq.py single true 20 > /tmp/yahoo_click_train.txt

echo 'generate /tmp/yahoo_click_train.txt'

# head -n 10000 yahoo_dataset/yahoo_nosiy_valid.txt | python generate_scan_seq.py single false 20 > /tmp/yahoo_click_valid.txt
# 
# echo 'generate /tmp/yahoo_click_valid.txt'
# 
# cat yahoo_dataset/yahoo_nosiy_test.txt | python generate_scan_seq.py single false 20 > /tmp/yahoo_click_test.txt
# 
# echo 'generate /tmp/yahoo_click_test.txt'
