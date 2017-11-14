/*************************************************
 *
 * Author: ananliu
 * Create time: 2014 Jun 10 15:49:30
 * E-Mail: ananliu@tencent.com
 * version 1.1
 *
*************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <algorithm>

#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>

#include "lda_trainer.h"

using namespace std;

void test_dataset(char* trndata, char* map)
{
    Dataset set;
    string dfile(trndata);
    string wordmap(map);
    set.read_trndata(dfile, wordmap);
}
void test_train(uint32_t C, const std::string& dfile, uint32_t K, uint32_t niters, uint32_t savestep, double alpha, double beta, uint32_t twords, bool flag)
{
    LDATrainer* trainer = new LDATrainer;
    trainer->train(C, dfile, K, niters, savestep, alpha, beta, twords, flag);
    delete trainer;
}
int main(int argc, char** argv)
{
    if (argc == 3)
    {
        test_dataset(argv[1], argv[2]);
    }
    if (argc == 10)
    {
        uint32_t C = atoi(argv[1]);
        if (C < 1)
        {
            cerr << "thread number error" << endl;
            return 0;
        }
        string dfile(argv[2]);
        uint32_t K = atoi(argv[3]);
        uint32_t niters = atoi(argv[4]);
        uint32_t savestep = atoi(argv[5]);
        double alpha = atof(argv[6]);
        double beta = atof(argv[7]);
        uint32_t twords = atof(argv[8]);
        string word_map_flag = argv[9];
        bool flag = false;
        if (word_map_flag == "1"
                || word_map_flag == "true"
                || word_map_flag == "TRUE"
                || word_map_flag == "True")
        {
            flag = true;
        }
        test_train(C, dfile, K, niters, savestep, alpha, beta, twords, flag);
    }
    return 0;
}
