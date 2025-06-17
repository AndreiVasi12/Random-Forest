#include "randomforest.h"
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <vector>
RandomForest:: RandomForest(int numtrees,int maxDepth,int minSamples):
numtrees(numtrees), maxDepth(maxDepth), minSamples(minSamples){

}

void RandomForest::train(const std::vector<std::vector<double>> &X,const std::vector<int>& y){

    std::mt19937 rng(42);
    std::uniform_int_distribution<> dist(0,X.size()-1);
    DecTree treeTrainer;

    for(int i=0 ;i < numtrees ;i++){
        std::vector<std::vector<double>> X_sample;
        std::vector<int> y_sample;
        for(size_t j=0 ; j<X.size();j++){
            int idx=dist(rng);
            X_sample.push_back(X[idx]);
            y_sample.push_back(y[idx]);

        }
        Tree* tree = treeTrainer.train(X_sample, y_sample, maxDepth, minSamples);
        trees.push_back(tree);
    }
    
}

int RandomForest::predict(const std::vector<double>& x) {
    std::unordered_map<int, int> votes;
    DecTree treeTrainer;
    for (auto tree : trees) {
        int pred = treeTrainer.predict(tree, x);
        votes[pred]++;
    }

    int majority = -1, maxVotes = 0;
    for (auto& [label, count] : votes) {
        if (count > maxVotes) {
            maxVotes = count;
            majority = label;
        }
    }
    return majority;
}


