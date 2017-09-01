# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. 
#

#
# Download and tokenize data with PTB tokenizer
#

data_path=senteval_data
preprocess_exec=./tokenizer.sed

mkdir $data_path

DIS='https://cocolab.stanford.edu/datasets/discourse_task_v1.zip'



### download DIS
mkdir $data_path/DIS
curl -Lo $data_path/DIS/discourse_task_v1.zip $DIS
unzip $data_path/DIS/discourse_task_v1.zip -d $data_path/DIS
rm $data_path/DIS/discourse_task_v1.zip
rm -r $data_path/DIS/__MACOSX
mv $data_path/DIS/discourse_task_v1/* $data_path/DIS
rm -r $data_path/DIS/discourse_task_v1/

