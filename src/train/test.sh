#! /bin/bash
############################################
#
# Author: ananliu
# Create time: 2014 Jun 10 16:00:58
# E-Mail: liuanan123@qq.com
# version 1.1
#
############################################

#../blade-bin/train/lda_test traindata wordmap.txt

#void test_train(uint32_t C, const std::string& dfile, uint32_t K, uint32_t niters, uint32_t savestep, double alpha, double beta, uint32_t twords, bool flag)
# 线程数量 输入文件 topic数目 迭代次数 每多少轮保存一次模型 超参数（建议-1） 超参数（建议0.1） 每个topic下展示的词数目 false
#nohup ../blade-bin/train/lda_test 3 ./test_data/input.txt 8 1000 0 -1 0.1 20 false > ./test_data/log.txt 2>&1 &

# for g++
g++ -o test -Wall -lpthread  test.cpp dataset.cpp lda_trainer.cpp
nohup ./test 3 ../../train_data/input.txt 6 1000 0 -1 0.1 20 false > ../../train_data/log.txt 2>&1 &
