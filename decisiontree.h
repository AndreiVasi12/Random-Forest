#pragma once
#include <vector>

struct Tree {
    bool isLeaf;
    int label;

    int featureIndex;
    double threshold;

    Tree* left;
    Tree* right;

    Tree() : isLeaf(false), label(-1), featureIndex(-1), threshold(0.0), left(nullptr), right(nullptr) {}
};

class DecTree {
public:
    Tree* train(const std::vector<std::vector<double>>& X,
                const std::vector<int>& y,
                int maxDepth,
                int minSamplesSplit);

    int predict(Tree* node, const std::vector<double>& x);

private:
    Tree* buildTree(const std::vector<std::vector<double>>& X,
                    const std::vector<int>& y,
                    int depth,
                    int maxDepth,
                    int minSamplesSplit);

    int majorityClass(const std::vector<int>& y);
};
