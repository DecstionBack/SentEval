# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. 
#
#

data_path=senteval_data
preprocess_exec="sed -f tokenizer.sed"

mkdir $data_path

## --------- SemEVA ABSA SPANISH and CHINESE ---------

# download SemEval ABSA_CH and ABSA_SP manually
# from http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools
# (requires login)
# Subtask 1, Part B
python process_ABSA_SP.py
python process_ABSA_CH.py

## --------- STS SPANISH ------------------------------

# for STSEval format, no training, just test
# for STSBenchmarkEval format, supervised

STS17_SP_test_input='http://alt.qcri.org/semeval2017/task1/data/uploads/sts2017.eval.v1.1.zip'
# STS2017.eval.v1.1/STS.input.track3.es-es.txt
# 250 tab separated pairs
STS17_SP_test_output='http://alt.qcri.org/semeval2017/task1/data/uploads/sts2017.gs.zip'
# STS2017.gs/STS.gs.track3.es-es.txt
# 250 similarity scores, 1 per line

STS15_SP_test='http://ixa2.si.ehu.es/stswiki/images/9/9d/STS2015-es-test.zip'
# STS2015-es-test/STS.(input|gs).(newswire|wikipedia).txt
# wikipedia: 251 ; newswire: 500

STS14_SP_test='http://ixa2.si.ehu.es/stswiki/images/9/9a/STS2014-es-test.zip'
# sts-es-test.2014/STS.input.wikipedia.txt
# sts-es-test.2014/STS.gs.wikipedia.txt
# 324 tab separated pairs / similarity scores
# sts-es-test.2014/STS.input.news.txt
# sts-es-test.2014/STS.gs.news.txt
# 480 tab separated pairs / similarity scores

STS14_SP_trial='http://ixa2.si.ehu.es/stswiki/images/c/c4/STS2014-es-trial.zip'
# STS-Es-trial/sts.input.li65.txt
# 65 tab separated pairs
# STS-Es-trial/sts.gs.li65.txt
# 65 similarity scores, 1 per line

STS_paths=("$STS14_SP_trial" "$STS14_SP_test" "$STS15_SP_test" "$STS17_SP_test_input $STS17_SP_test_output")

mkdir $data_path/STS_SP

for task in 0 1 2 3;
do
    for fpath in ${STS_paths[$task]};
        do
        curl -Lo $data_path/STS_SP/data_$task.zip $fpath
        unzip $data_path/STS_SP/data_$task.zip -d $data_path/STS_SP/data_$task
        rm $data_path/STS_SP/data_$task.zip
    done
done

data_dirs=("data_0/STS-Es-trial" "data_1" "data_2" "data_3/STS2017.eval.v1.1" "data_3/STS2017.gs")
new_dirs=("STS14" "STS14" "STS15" "STS17" "STS17")

for i in 0 1 2 3 4;
do
    new_dir=$data_path/STS_SP/${new_dirs[$i]}
    mkdir $new_dir
    data_dir=$data_path/STS_SP/${data_dirs[$i]}
    mv $data_dir/*.input.* $new_dir/
    mv $data_dir/*.gs.* $new_dir/
done

for task in 0 1 2 3;
do
    for fpath in ${STS_paths[$task]};
        do
        rm -rf $data_path/STS_SP/data_$task
        # remove any 2017 tasks with english or arabic:
        rm $data_path/STS_SP/STS17/*en*
        rm $data_path/STS_SP/STS17/*ar*
    done
done

mv senteval_data/STS_SP/STS14/sts.gs.li65.txt senteval_data/STS_SP/STS14/STS.gs.li65.txt
mv senteval_data/STS_SP/STS14/sts.input.li65.txt senteval_data/STS_SP/STS14/STS.input.li65.txt

python process_STS_SP.py

## could create alternative supervised version from this.
## STSBenchmark has 5749 training examples, 1379 test, and 1500 dev
## we only have 1869 total examples

## --------- DIS ENGLISH, SPANISH, and CHINESE ---------

DIS='https://cocolab.stanford.edu/datasets/discourse_task_v1.zip'
DIS_SP=''
DIS_CH=''

# ### download DIS
# mkdir $data_path/DIS
# curl -Lo $data_path/DIS/discourse_task_v1.zip $DIS
# unzip $data_path/DIS/discourse_task_v1.zip -d $data_path/DIS
# rm $data_path/DIS/discourse_task_v1.zip
# rm -r $data_path/DIS/__MACOSX
# mv $data_path/DIS/discourse_task_v1/* $data_path/DIS
# rm -r $data_path/DIS/discourse_task_v1/

## --------- SAB SPANISH ------------------------------

# # CR
# mkdir $data_path/CR
# cat $data_path/data_bin_classif/data/customerr/custrev.pos | $preprocess_exec > $data_path/CR/custrev.pos
# cat $data_path/data_bin_classif/data/customerr/custrev.neg | $preprocess_exec > $data_path/CR/custrev.neg
