#include "lda_trainer.h"

#include <memory.h>
#include <stdlib.h>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <vector>
#include <assert.h>

Dataset* LDATrainer::s_trndata;
int** LDATrainer::s_nw;
int** LDATrainer::s_nwsum;
int** LDATrainer::s_z;
int** LDATrainer::s_nd;
int* LDATrainer::s_ndsum;
int*** LDATrainer::s_delta_nw;
int LDATrainer::s_C;
int LDATrainer::s_M;
int LDATrainer::s_V;
int LDATrainer::s_K;
int LDATrainer::s_twords;
int LDATrainer::s_savestep;
int LDATrainer::s_niters;
double LDATrainer::s_alpha, LDATrainer::s_beta;
double LDATrainer::s_Kalpha, LDATrainer::s_Vbeta;
std::string LDATrainer::s_dir;

pthread_mutex_t LDATrainer::map_lock;
pthread_mutex_t LDATrainer::reduce_lock;
pthread_cond_t LDATrainer::map_ready;
pthread_cond_t LDATrainer::reduce_ready;
int LDATrainer::map_ready_num;

pthread_mutex_t LDATrainer::log_lock;

LDATrainer::LDATrainer()
{
}

LDATrainer::LDATrainer(int c)
{
    init_local_data(c);
}

LDATrainer::~LDATrainer()
{
    uninit_local_data();
}

void LDATrainer::train(int C, const std::string& dfile, int K, int niters, int savestep, double alpha, double beta, int twords, bool flag)
{
    int idx = dfile.size() - 1;
    for (;idx >= 0; --idx)
    {
        if (dfile[idx] == '/')
        {
            break;
        }
    }
    if (idx < 0)
    {
        s_dir = "./";
    }
    else
    {
        s_dir = dfile.substr(0, idx+1);
    }
    std::string wordmapfile = s_dir+"wordmap.txt";
    init(C, dfile, wordmapfile, K, niters, savestep, alpha, beta, twords, flag);
    init_local_data(C);
    pthread_t* pt = new pthread_t[C];
    int* pid = new int[C];

    for (int i = 0; i < C; ++i)
    {
        pid[i] = i;
        pthread_create(&pt[i], NULL, &mapper, &pid[i]);
    }
    reduce();
    for (int i = 0; i < C; ++i)
    {
        pthread_join(pt[i], NULL);
    }
    uninit();
    delete[] pt;
    delete[] pid;
}

void* LDATrainer::mapper(void* mapper_id)
{
    int i = *(int *) mapper_id;
    LDATrainer* m = new LDATrainer(i);
    m->map();
    delete m;
    return ((void *) 0);
}

void LDATrainer::map()
{
    char buf[1024];
    std::string log;
    sprintf(buf, "mapper %u: begin!", c);
    log = buf;
    print(log);
    for (int i = 0; i < niters; ++i)
    {
        sprintf(buf, "mapper %u iteration %u begin to sampling!", c, (i+1));
        log = buf;
        print(log);
        naive_map();        // for sampling
        sprintf(buf, "mapper %u iteration %u done!", c, (i+1));
        log = buf;
        print(log);
        pthread_mutex_lock(&map_lock);
        ++map_ready_num;
        pthread_cond_signal(&map_ready);
        
        // waiting for reduce
        pthread_mutex_lock(&reduce_lock);
        pthread_mutex_unlock(&map_lock);
        pthread_cond_wait(&reduce_ready, &reduce_lock);
        pthread_mutex_unlock(&reduce_lock);
    }
    sprintf(buf, "mapper %u done", c);
    log = buf;
    print(log);
}

void LDATrainer::reduce()
{
    char buf[1024];
    std::string log;
    for (int i = 0; i < niters; ++i)
    {
        // waiting for map
        pthread_mutex_lock(&map_lock);
        while (map_ready_num != s_C)
        {
            pthread_cond_wait(&map_ready, &map_lock);
        }
        pthread_mutex_unlock(&map_lock);
        
        pthread_mutex_lock(&reduce_lock);
        sprintf(buf, "reducer: iteration %u begin to reduce!", (i+1));
        log = buf;
        print(log);
        naive_reduce();
        sprintf(buf, "reducer: iteration %u done!", (i+1));
        log = buf;
        print(log);
        if (s_savestep > 0 && i != niters-1 && (i+1)%s_savestep == 0)
        {
            int save_iter = i+1;
            char buf[64];
            sprintf(buf, "%05d", save_iter);
            std::string pre = s_dir+"model-"+buf;
            save_model(pre);
        }
        pthread_cond_broadcast(&reduce_ready);  // broad signal to all mappers
        map_ready_num = 0;
        pthread_mutex_unlock(&reduce_lock);
    }
    std::string pre = s_dir+"model-final";
    save_model(pre);
}

int LDATrainer::init(int C, const std::string& dfile, const std::string& wordmapfile, int K, int niters, int savestep, double alpha, double beta, int twords, bool flag)
{
    s_C = C;
    s_K = K;
    s_niters = niters;
    s_savestep = savestep;
    if (twords < 20)
    {
        s_twords = 20;
    }
    else
    {
        s_twords = twords;
    }
    if (alpha <= 0)
    {
        s_alpha = 50.0 / K;
    }
    else
    {
        s_alpha = alpha;
    }
    if (beta <= 0)
    {
        s_beta = 0.1;
    }
    else
    {
        s_beta = beta;
    }
    s_trndata = new Dataset();
    int ret = s_trndata->read_trndata(dfile, wordmapfile, flag);
    if (ret)
    {
        return ret;
    }
    s_M = s_trndata->m_M;
    s_V = s_trndata->m_V;
    
    s_Kalpha = s_alpha * s_K;
    s_Vbeta = s_beta * s_V;
    ret = new_arr(s_nw, s_V, s_K);
    if (ret != 0)
    {
        return ret;
    }
    my_memset(s_nw, s_V, s_K);

    s_z = new int*[s_M];
    for (int i = 0; i < s_M; ++i)
    {
        s_z[i] = new int[s_trndata->m_docs[i]->m_length];
        memset(s_z[i], 0, sizeof(int) * s_trndata->m_docs[i]->m_length);
    }
    
    ret = new_arr(s_nd, s_M, s_K);
    if (ret)
    {
        return ret;
    }
    my_memset(s_nd, s_M, s_K);

    ret = new_arr(s_delta_nw, s_C+1, s_V, s_K);
    if (ret)
    {
        return ret;
    }
    my_memset(s_delta_nw, s_C+1, s_V, s_K);
    /*
    s_nwsum = new int[s_K];
    memset(s_nwsum, 0, s_K * sizeof(int));
    */
    s_ndsum = new int[s_M];
    memset(s_ndsum, 0, s_M * sizeof(int));
    ret = new_arr(s_nwsum, s_C+1, s_K);
    my_memset(s_nwsum, s_C+1, s_K);
    if (ret)
    {
        return ret;
    }
    int topic = 0;
    srandom(time(0)+pthread_self());
    for (int m = 0; m < s_M; ++m)
    {
        for (uint32_t n = 0; n < s_trndata->m_docs[m]->m_length; ++n)
        {
            //topic = (int) (((double) random()) / RAND_MAX * s_K);
            if (topic == s_K)
            {
                topic = 0;
            }
            s_z[m][n] = topic;
            ++s_nw[s_trndata->m_docs[m]->m_words[n]][topic];
            ++s_nd[m][topic];
            ++s_nwsum[s_C][topic];
            ++topic;
        }
        s_ndsum[m] = s_trndata->m_docs[m]->m_length;
    }
    my_memcpy(s_C, s_C);    // copy s_nw to s_delta_nw[s_C]
    pthread_mutex_init(&map_lock, NULL);
    pthread_mutex_init(&reduce_lock, NULL);
    pthread_cond_init(&map_ready, NULL);
    pthread_cond_init(&reduce_ready, NULL);
    map_ready_num = 0;

    pthread_mutex_init(&log_lock, NULL);
    return 0;
}

void LDATrainer::uninit()
{
    delete s_trndata;
    delete_arr(s_nw);
    if (s_z != NULL)
    {
        for (int i = 0; i < s_M; ++i)
        {
            if (s_z[i] != NULL)
            {
                delete[] s_z[i];
                s_z[i] = NULL;
            }
        }
        delete[] s_z;
        s_z = NULL;
    }
    delete_arr(s_nwsum);
    delete_arr(s_nd);
    delete_arr(s_delta_nw, s_C+1);
    pthread_cond_destroy(&map_ready);
    pthread_cond_destroy(&reduce_ready);
}

void LDATrainer::init_local_data(int c)
{
    this->c = c;
    M = s_M;
    V = s_V;
    K = s_K;
    niters = s_niters;
    alpha = s_alpha;
    beta = s_beta;
    Kalpha = s_Kalpha;
    Vbeta = s_Vbeta;
    
    /*
    if (c == s_C) // reduce thread
    {
        begin_docid = 0;
        end_docid = s_M;
    }
    else
    {
        int sub_num = (s_M + s_C - 1) / s_C;
        begin_docid = sub_num * c;
        end_docid = sub_num * (c + 1);
        if (c == s_C - 1)
        {
            end_docid = s_M;
        }
    }
    std::string log;
    char buf[256];
    sprintf(buf, "thread %u: doc%u to doc%u", c, begin_docid, end_docid-1);
    log = buf;
    print(log);
    */
    p = new double[K];
    srandom(time(0) + pthread_self());
}

void LDATrainer::uninit_local_data()
{
    if (p != NULL)
    {
        delete[] p;
        p = NULL;
    }
}

void LDATrainer::naive_reduce()
{
    int32_t i, k, v;
    memset(s_nwsum[s_C], 0, s_K * sizeof(int));
    for (i = 0; i < s_C; ++i)
    {
        for (v = 0; v < V; ++v)
        {
            for (k = 0; k < K; ++k)
            {
                s_nw[v][k] += (s_delta_nw[i][v][k] - s_delta_nw[s_C][v][k]);
            }
        }
    }
    for (k = 0; k < K; ++k)
    {
        for (v = 0; v < V; ++v)
        {
            s_nwsum[s_C][k] += s_nw[v][k];
        }
    }
    my_memcpy(s_C, s_C);    // copy s_nw to s_delta_nw[s_C]
}

void LDATrainer::naive_map()
{
    //my_memset(s_delta_nw[c], V, K);
    my_memcpy(c, s_C);
    int m;
    uint32_t n;
    //for (m = begin_docid; m < end_docid; ++m)
    for (m = c; m < s_M; m += s_C)
    {
        for (n = 0; n < s_trndata->m_docs[m]->m_length; ++n)
        {
            sampling(m, n);
        }
    }
}

int LDATrainer::sampling(int m, int n)
{
    int topic = s_z[m][n];
    int w = s_trndata->m_docs[m]->m_words[n];
    // remove z_i from the count variables
    --s_delta_nw[c][w][topic];
    --s_nd[m][topic];
    --s_nwsum[c][topic];
    
    int k;
    for (k = 0; k < K; ++k)
    {
        p[k] = (s_delta_nw[c][w][k] + beta) / (s_nwsum[c][k] + Vbeta) *
            (s_nd[m][k] + alpha) / (s_ndsum[m] - 1 + Kalpha);
        if (k)
        {
            p[k] += p[k-1];
        }
    }
    double sample_p = ((double)random() / RAND_MAX) * p[K - 1];
    int begin = 0; 
    int end = K; 
    k = (begin + end) / 2; 
    while (sample_p != p[k] && begin < end) 
    {    
        if (sample_p >= p[k])
        {    
            begin = k+1;
        }    
        else 
        {    
            end = k; 
        }    
        k = (begin+end)/2;
    }    
    while (sample_p >= p[k] && k != K-1) 
    {    
        ++k;
    }
    assert((sample_p < p[k] || k == K-1) && (k == 0 || sample_p > p[k-1]));
    ++s_delta_nw[c][w][k];
    ++s_nd[m][k];
    ++s_nwsum[c][k];
    s_z[m][n] = k;
    return k;
}

int LDATrainer::new_arr(int**& p, int m, int n)
{
    p = new int*[m];
    p[0] = new int[m * n];
    for (int i = 1; i < m; i++)
    {
        p[i] = p[i-1] + n;
    }
    return 0;
}

int LDATrainer::new_arr(int***& p, int k, int m, int n)
{
    p = new int**[k];
    int ret;
    for (int i = 0; i < k; ++i)
    {
        ret = new_arr(p[i], m, n);
        if (ret)
        {
            return ret;
        }
    }
    return 0;
}

int LDATrainer::delete_arr(int**& p)
{
    if (p != NULL)
    {
        if (p[0] != NULL)
        {
            delete[] p[0];
            p[0] = NULL;
        }
        delete[] p;
        p = NULL;
    }
    return 0;
}

int LDATrainer::delete_arr(int***& p, int k)
{
    int ret;
    if (p != NULL)
    {
        for (int i = 0; i < k; ++i)
        {
            ret = delete_arr(p[i]);
            if (ret != 0)
            {
                return ret;
            }
        }
        delete[] p;
        p = NULL;
    }
    return 0;
}

int LDATrainer::my_memset(int**& p, int m, int n)
{
    memset(p[0], 0, m * n * sizeof(int));
    return 0;
}

int LDATrainer::my_memset(int***& p, int k, int m, int n)
{
    int ret;
    for (int i = 0; i < k; ++i)
    {
        ret = my_memset(p[i], m, n);
        if (ret != 0)
        {
            return 0;
        }
    }
    return 0;
}
int LDATrainer::save_model(const std::string& pre)
{
    return save_model_tassign(pre) +
        save_model_theta(pre) +
        save_model_phi(pre) +
        save_model_others(pre);
}

int LDATrainer::save_model_tassign(const std::string& pre)
{
    std::string file = pre + ".tassign";
    std::ofstream fout(file.c_str());
    if (!fout.is_open())
    {
        std::cerr << "ERROR: Can't open file '" << file << "'!" << std::endl;
        return -1;
    }
    int i;
    uint32_t j;
    for (i = 0; i < s_M; ++i)
    {
        for (j = 0; j < s_trndata->m_docs[i]->m_length; ++j)
        {
            fout << s_trndata->m_docs[i]->m_words[j] << ":" << s_z[i][j] << " ";
        }
        fout << std::endl;
    }
    fout.close();
    return 0;
}
int LDATrainer::save_model_theta(const std::string& pre)
{
    std::string file = pre + ".theta";
    std::ofstream fout(file.c_str());
    if (!fout.is_open())
    {
        std::cerr << "ERROR: Can't open file '" << file << "'!" << std::endl;
        return -1;
    }
    int m, k;
    double ptheta;
    for (m = 0; m < s_M; ++m)
    {
        for (k = 0; k < s_K; ++k)
        {
            ptheta = (s_nd[m][k] + s_alpha) / (s_ndsum[m] + s_Kalpha);
            fout << ptheta << " ";
        }
        fout << std::endl;
    }
    fout.close();
    return 0;
}
int LDATrainer::save_model_phi(const std::string& pre)
{
    std::string file = pre + ".phi";
    std::string tfile = pre + ".twords";
    std::ofstream fout(file.c_str());
    std::ofstream ftopic(tfile.c_str());
    if (!fout.is_open())
    {
        std::cerr << "ERROR: Can't open file '" << file << "'!" << std::endl;
        return -1;
    }
    int k, v;
    size_t i;
    double pphi;
    std::vector< std::pair<int, double> > topic_words;
    std::pair<int, double> word_prob;
    for (k = 0; k < s_K; ++k)
    {
        for (v = 0; v < s_V; ++v)
        {
            pphi = (s_nw[v][k] + s_beta) / (s_nwsum[s_C][k] + s_Vbeta);
            fout << pphi << " ";
            word_prob.first = v;
            word_prob.second = pphi;
            topic_words.push_back(word_prob);
        }
        fout << std::endl;
        stable_sort(topic_words.begin(), topic_words.end(), my_cmp);
        ftopic << "Topic " << k << "th:" << std::endl;
        for (i = 0; i < (size_t) s_twords && i < topic_words.size(); ++i)
        {
            ftopic << "\t" << (*(s_trndata->m_id2word))[topic_words[i].first] << "   " << topic_words[i].second << std::endl;
        }
        topic_words.clear();
    }
    fout.close();
    ftopic.close();
    return 0;
}
int LDATrainer::save_model_others(const std::string& pre)
{
    std::string file = pre + ".others";
    std::ofstream fout(file.c_str());
    if (!fout.is_open())
    {
        std::cerr << "ERROR: Can't open file '" << file << "'!" << std::endl;
        return -1;
    }
    fout << "alpha=" << s_alpha << std::endl;
    fout << "beta=" << s_beta << std::endl;
    fout << "ntopics=" << s_K << std::endl;
    fout << "ndocs=" << s_M << std::endl;
    fout << "nwords=" << s_V << std::endl;
    fout << "liter=" << s_niters << std::endl;
    fout.close();
    return 0;
}
/*
int LDATrainer::save_model_twords(const std::string& pre)
{
    std::string file = pre + ".twords";
    std::ofstream fout(file.c_str());
    if (!fout.is_open())
    {
        std::cerr << "ERROR: Can't open file '" << file << "'!" << std::endl;
        return -1;
    }
    return 0;
}
*/

bool LDATrainer::my_cmp(const std::pair<int, double>& a, const std::pair<int, double>& b)
{
    return a.second > b.second;
}
void LDATrainer::print(const std::string& log)
{
    pthread_mutex_lock(&log_lock);
    std::cout << log << std::endl;
    pthread_mutex_unlock(&log_lock);
}

int LDATrainer::my_memcpy(int child_id, int father_id)
{
    memcpy(s_delta_nw[child_id][0], s_nw[0], s_K * s_V * sizeof(int));
    if (father_id != child_id)
    {
        memcpy(s_nwsum[child_id], s_nwsum[father_id], s_K * sizeof(int));
    }
    return 0;
}
