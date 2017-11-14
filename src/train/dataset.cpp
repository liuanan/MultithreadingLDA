#include "dataset.h"

#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>

Document::Document(uint32_t length, uint32_t* words)
{
    init(length, words);
}

Document::Document(const std::vector<uint32_t>& words)
{
    if (words.size() != 0)
    {
        init(words.size(), (const uint32_t*) &words[0]);
    }
}

Document::~Document()
{
    uninit();
}

int Document::init(uint32_t length, const uint32_t* words)
{
    m_length = length;
    if (length != 0)
    {
        assert(words != NULL);
        m_words = new uint32_t[m_length];
        memcpy(m_words, words, length*sizeof(uint32_t));
    }
    return 0;
}

int Document::uninit()
{
    if (m_words != NULL)
    {
        delete[] m_words;
        m_words = NULL;
        m_length = 0;
    }
    return 0;
}

Dataset::Dataset(int M)
{
    m_M = M;
    m_V = 0;
    m_docs = NULL;
    m_word2id = NULL;
    m_id2word = NULL;
}

Dataset::~Dataset()
{
    deleteDataset();
}

int Dataset::read_trndata(const std::string& dfile, const std::string& wordmapfile, bool flag)
{
    if (flag)
    {
        return read_trndata_with_wordmapfile(dfile, wordmapfile);
    }
    return read_trndata_without_wordmapfile(dfile, wordmapfile);
}

int Dataset::read_trndata_with_wordmapfile(const std::string& dfile, const std::string& wordmapfile)
{
    int ret = read_wordmap(wordmapfile);
    if (ret)
    {
        return ret;
    }
    std::ifstream fin(dfile.c_str());
    if (!fin.is_open())
    {
        std::cerr << "ERROR: file " << dfile << " not exists!" << std::endl;
        return -1;
    }
    std::string line;
    if (getline(fin, line))
    {
        m_M = atoi(line.c_str());
        if (m_M == 0)
        {
            std::cerr << "ERROR: first line of trndata file!" << std::endl;
            return -1;
        }
        m_docs = new Document*[m_M]; 
        for (uint32_t i = 0; i < m_M; ++i)
        {
            m_docs[i] = NULL;
        }
    }
    else
    {
        std::cerr << "ERROR: empty trndata file!" << std::endl;
        return -2;
    }
    uint32_t M = 0;
    std::string term;
    std::vector<uint32_t> words;
    mapword2id::iterator iter;
    while (getline(fin, line))
    {
        if (M >= m_M)
        {
            ++M;
            break;
        }
        words.clear();
        std::istringstream istring(line);
        while (istring >> term)
        {
            iter = m_word2id->find(term);
            if (iter == m_word2id->end())
            {
                std::cerr << "WARM: term '" << term << "'not in wordmap file!" << std::endl;
                continue;
            }
            words.push_back(iter->second);
        }
        if (words.size() == 0)
        {
            std::cerr << "ERROR: empty document in trndata, line: " << (M+2) << std::endl;
            fin.close();
            return -1;
        }
        m_docs[M] = new Document(words);
        ++M;
    }
    fin.close();
    if (M != m_M)
    {
        std::cerr << "ERROR: trndata number" << std::endl;
        return -1;
    }
    m_V = m_word2id->size();
    return 0;
}

int Dataset::read_trndata_without_wordmapfile(const std::string& dfile, const std::string& wordmapfile)
{
    m_word2id = new mapword2id;
    m_id2word = new mapid2word;
    std::ifstream fin(dfile.c_str());
    if (!fin.is_open())
    {
        std::cerr << "ERROR: file " << dfile << " not exists!" << std::endl;
        return -1;
    }
    std::string line;
    if (getline(fin, line))
    {
        m_M = atoi(line.c_str());
        if (m_M == 0)
        {
            std::cerr << "ERROR: first line of trndata file!" << std::endl;
            return -1;
        }
        m_docs = new Document*[m_M]; 
        for (uint32_t i = 0; i < m_M; ++i)
        {
            m_docs[i] = NULL;
        }
    }
    else
    {
        std::cerr << "ERROR: empty trndata file!" << std::endl;
        return -2;
    }
    uint32_t M = 0;
    std::string term;
    std::vector<uint32_t> words;
    mapword2id::const_iterator iter;
    uint32_t term_cnt = 0;
    while (getline(fin, line))
    {
        if (M >= m_M)
        {
            ++M;
            break;
        }
        words.clear();
        std::istringstream istring(line);
        while (istring >> term)
        {
            iter = m_word2id->find(term);
            if (iter == m_word2id->end())
            {
                (*m_word2id)[term] = term_cnt;
                words.push_back(term_cnt);
                (*m_id2word)[term_cnt] = term;
                ++term_cnt;
            }
            else
            {
                words.push_back(iter->second);
            }
        }
        if (words.size() == 0)
        {
            std::cerr << "ERROR: empty document in trndata, line: " << (M+2) << std::endl;
            fin.close();
            return -1;
        }
        m_docs[M] = new Document(words);
        ++M;
    }
    fin.close();
    if (M != m_M)
    {
        std::cerr << "ERROR: trndata number" << std::endl;
        return -1;
    }
    m_V = m_word2id->size();
    return write_wordmap(wordmapfile);
}

int Dataset::write_wordmap(const std::string& wordmapfile) const
{
    std::ofstream fout(wordmapfile.c_str());
    if (!fout.is_open())
    {
        std::cerr << "ERROR: write wordmap file '" << wordmapfile << "'!" << std::endl;
        return -1;
    }
    assert(m_word2id);
    fout << m_word2id->size() << std::endl;
    for (mapword2id::const_iterator iter = m_word2id->begin(); iter != m_word2id->end(); ++iter)
    {
        fout << iter->first << " " << iter->second << std::endl;
    }
    fout.close();
    return 0;
}

int Dataset::read_wordmap(const std::string& wordmapfile)
{
    m_word2id = new mapword2id;
    m_id2word = new mapid2word;
    std::ifstream fin(wordmapfile.c_str());
    if (!fin.is_open())
    {
        std::cerr << "ERROR: open wordmap file '" << wordmapfile << "'!" << std::endl;
        return -1;
    }
    std::string line;
    if (!getline(fin, line))
    {
        std::cerr << "ERROR: empty file " << wordmapfile << "!" << std::endl;
        fin.close();
        return -1;
    }
    m_V = atoi(line.c_str());
    if (m_V == 0)
    {
        std::cerr << "ERROR: wrong input in first line of file '" << wordmapfile << "'!" << std::endl;
        fin.close();
        return -1;
    }
    std::string term;
    uint32_t id;
    uint32_t term_cnt = 0;
    while (getline(fin, line))
    {
        std::istringstream istring(line);
        if (istring >> term >> id)
        {
            (*m_word2id)[term] = id;
            (*m_id2word)[id] = term;
        }
        else
        {
            std::cerr << "ERROR: wrong input in line " << (term_cnt+2) << " of file '" << wordmapfile << "'!" << std::endl;
            fin.close();
            return -1;
        }
        ++term_cnt;
    }
    fin.close();
    return 0;
}

void Dataset::deleteDataset()
{
    if (m_docs != NULL)
    {
        for (uint32_t i = 0; i < m_M; ++i)
        {
            if (m_docs[i] != NULL)
            {
                delete m_docs[i];
            }
            m_docs[i] = NULL;
        }
        delete[] m_docs;
        m_docs = NULL;
    }
    if (m_word2id != NULL)
    {
        delete m_word2id;
        m_word2id = NULL;
    }
    if (m_id2word != NULL)
    {
        delete m_id2word;
        m_id2word = NULL;
    }
}

