#!/usr/bin/env bash

cat ~/Data/yahoo_c14_ltr/yahoo_nosiy_train.txt | python2 generate_scan_seq.py seq true 20 > /tmp/yahoo_seq_train.txt

echo 'generate /tmp/yahoo_seq_train.txt'

cat ~/Data/yahoo_c14_ltr/yahoo_nosiy_valid.txt | python2 generate_scan_seq.py seq false 20 > /tmp/yahoo_seq_valid.txt

echo 'generate /tmp/yahoo_seq_valid.txt'

 cat ~/Data/yahoo_c14_ltr/yahoo_nosiy_test.txt | python2 generate_scan_seq.py seq false 20 > /tmp/yahoo_seq_test.txt

 echo 'generate /tmp/yahoo_seq_test.txt'
