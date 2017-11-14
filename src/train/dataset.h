/*************************************************
 *
 * Author: ananliu
 * Create time: 2014.06.14 12:10:22
 * E-Mail: liuanan123@qq.com
 * version 1.1
 *
*************************************************/

#ifndef _DATA_SET_H_
#define _DATA_SET_H_

#include <vector>
#include <string>
#include <map>
#include <stdint.h>

typedef std::map<std::string, uint32_t> mapword2id;
typedef std::map<uint32_t, std::string> mapid2word;

class Document
{
public:
    Document(uint32_t length=0, uint32_t* words=NULL);
    Document(const std::vector<uint32_t>& words);
    ~Document();

private:
    int init(uint32_t length, const uint32_t* words);
    int uninit();

public:
    uint32_t* m_words;
    uint32_t m_length;
};

class Dataset
{
public:
    Dataset(int M=0);
    ~Dataset();

    /**
     * dfile: raw data file name
     * wordmapfile: wordmap file name
     * flag: wordmapfile is pre-generated or not
     */
    int read_trndata(const std::string& dfile, const std::string& wordmapfile, bool flag=false);

private:    
    int read_trndata_with_wordmapfile(const std::string& dfile, const std::string& wordmapfile);
    int read_trndata_without_wordmapfile(const std::string& dfile, const std::string& wordmapfile);

    int write_wordmap(const std::string& wordmapfile) const;
    int read_wordmap(const std::string& wordmapfile);
    void deleteDataset();

public:
    Document ** m_docs;
    mapword2id* m_word2id;
    mapid2word* m_id2word;
    uint32_t m_M;
    uint32_t m_V;
};

#endif
