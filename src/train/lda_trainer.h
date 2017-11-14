/*************************************************
 *
 * Author: ananliu
 * Create time: 2014.06.14 11:58:11
 * E-Mail: liuanan123@qq.com
 * version 1.1
 *
*************************************************/

#ifndef _LDA_TRAINNER_H_
#define _LDA_TRAINNER_H_

#include <string>
#include <utility>
#include <unistd.h>
#include <pthread.h>

#include "dataset.h"

class LDATrainer
{
public:
    LDATrainer();
    LDATrainer(int c);
    ~LDATrainer();
    void train(int C, const std::string& dfile, int K, int niters, int savestep=0, double alpha=-1, double beta=-1, int twords=20, bool flag=false);

private:
    /**
     * inition of static data
     */
    static int init(int C, const std::string& dfile, const std::string& wordmapfile, int K, int niters, int savestep=0, double alpha=-1, double beta=-1, int twords=20, bool flag=false);
    static void uninit();

    void init_local_data(int c);
    void uninit_local_data();

    static void* mapper(void* lda_trainer);

    /**
     * 子线程的执行逻辑
     */
    void map();

    /**
     * 合并子线程的结果
     */
    void reduce();

    /**
     * single iteration
     */
    void naive_reduce();
    void naive_map();
    int sampling(int m, int n);
    static int new_arr(int**& p, int m, int n);
    static int new_arr(int***& p, int k, int m, int n);
    static int delete_arr(int**& p);
    static int delete_arr(int***& p, int k);
    static int my_memset(int**& p, int m, int n);
    static int my_memset(int***& p, int k, int m, int n);

    // save the lda model
    static int save_model(const std::string& pre);
    static int save_model_tassign(const std::string& pre);
    static int save_model_theta(const std::string& pre);
    static int save_model_phi(const std::string& pre);
    static int save_model_others(const std::string& pre);
    //static int save_model_twords(const std::string& pre);
    static bool my_cmp(const std::pair<int, double>& a, const std::pair<int, double>& b);
    static int my_memcpy(int child_id, int father_id);

    static void print(const std::string& log);

private:
    static Dataset* s_trndata;      // the training data

    static int** s_nw;         // s_nw[i][j]: number of instances of term i assigned to topic j, size s_V * s_K
    static int** s_nwsum;       // nwsum[c][j]: total number of words assigned to topic j, size s_C * s_K
    static int** s_z;          // topic assignments for words, size M * document.length; save in tassgin file
    static int** s_nd;         // nd[i][j]: number of words in document i assigned to topic j, size s_M * s_K
    static int* s_ndsum;       // nasum[i]: number of words in document i, size M
    static int*** s_delta_nw;  // s_delta_nw[c][i][j]: number of instances of term i assigned to topic j in thread c, size s_C * s_V * s_K

    static int s_C;            // number of map thread
    static int s_M;            // size of trainning data
    static int s_V;            // size of vocabulary
    static int s_K;            // size of topics
    static int s_twords;       // top words of each topic
    static int s_savestep;     // saving steps
    static int s_niters;       // iterations
    static double s_alpha, s_beta;  // hyperparameters of LDA model
    static double s_Kalpha, s_Vbeta;
    static std::string s_dir;    // data dir

private:
    int c;                     // thread id, [0, C]
    int M;                     // size of global training data
    int V;                     // size of vocabulary
    int K;                     // size of topics
    int niters;                // iterations
    double alpha, beta;        // hyperparameters of LDA model
    double Kalpha, Vbeta;
    double* p;                 // temp variable for sampling

private:
    static pthread_mutex_t map_lock;    // map锁
    static pthread_mutex_t reduce_lock; // reduce锁
    static pthread_cond_t map_ready;    // map的信号量
    static pthread_cond_t reduce_ready; // reduce的信号量
    static int map_ready_num;           // 本轮已完成的map个数
    static pthread_mutex_t log_lock;    // 日志文件写锁
};

#endif
