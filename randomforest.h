#pragma once
#include "decisiontree.h"
#include <random>
#include "csvreader.h"
#include <vector>

class RandomForest{
    private:
    int numtrees,maxDepth,minSamples;
    std::vector<Tree*> trees;
    public:
    RandomForest(int numtrees,int maxDepth,int minSamples);
    void train(const std::vector<std::vector<double>> &X,const std::vector<int>& y);
    int predict(const std::vector<double> &x);
    


};
