#include "decisiontree.h"
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <cmath>

int DecTree::majorityClass(const std::vector<int>& y) {
    std::unordered_map<int, int> counts;
    for (int label : y) {
        counts[label]++;
    }
    int majority = y[0], maxCount = 0;
    for (auto& kv : counts) {
        if (kv.second > maxCount) {
            maxCount = kv.second;
            majority = kv.first;
        }
    }
    return majority;
}

Tree* DecTree::train(const std::vector<std::vector<double>>& X,
                     const std::vector<int>& y,
                     int maxDepth,
                     int minSamplesSplit)
{
    return buildTree(X, y, 0, maxDepth, minSamplesSplit);
}

Tree* DecTree::buildTree(const std::vector<std::vector<double>>& X,
                         const std::vector<int>& y,
                         int depth,
                         int maxDepth,
                         int minSamplesSplit)
{
    Tree* node = new Tree();

    // Stop conditions
    if (depth >= maxDepth || y.size() < minSamplesSplit) {
        node->isLeaf = true;
        node->label = majorityClass(y);
        return node;
    }

    int bestFeature = -1;
    double bestThreshold = 0.0;
    double bestGini = std::numeric_limits<double>::max();
    std::vector<int> leftIdx, rightIdx;

    int numFeatures = X[0].size();

    // Try every feature and every sample's value as threshold
    for (int feature = 0; feature < numFeatures; ++feature) {
        std::vector<std::pair<double, int>> featureValues;
        for (int i = 0; i < X.size(); ++i) {
            featureValues.emplace_back(X[i][feature], y[i]);
        }

        std::sort(featureValues.begin(), featureValues.end());

        std::vector<int> leftLabels, rightLabels(y.begin(), y.end());

        for (int i = 1; i < featureValues.size(); ++i) {
            leftLabels.push_back(featureValues[i - 1].second);
            rightLabels.erase(rightLabels.begin());

            if (featureValues[i].first == featureValues[i - 1].first)
                continue;

            double giniLeft = 1.0, giniRight = 1.0;
            std::unordered_map<int, int> countL, countR;

            for (int label : leftLabels) countL[label]++;
            for (int label : rightLabels) countR[label++];

            giniLeft = 1.0;
            for (auto& kv : countL) {
                double p = (double)kv.second / leftLabels.size();
                giniLeft -= p * p;
            }

            giniRight = 1.0;
            for (auto& kv : countR) {
                double p = (double)kv.second / rightLabels.size();
                giniRight -= p * p;
            }

            double weightedGini = (leftLabels.size() * giniLeft + rightLabels.size() * giniRight) / y.size();

            if (weightedGini < bestGini) {
                bestGini = weightedGini;
                bestFeature = feature;
                bestThreshold = (featureValues[i].first + featureValues[i - 1].first) / 2;
            }
        }
    }

    if (bestFeature == -1) {
        node->isLeaf = true;
        node->label = majorityClass(y);
        return node;
    }

    // Split data
    std::vector<std::vector<double>> X_left, X_right;
    std::vector<int> y_left, y_right;

    for (int i = 0; i < X.size(); ++i) {
        if (X[i][bestFeature] < bestThreshold) {
            X_left.push_back(X[i]);
            y_left.push_back(y[i]);
        } else {
            X_right.push_back(X[i]);
            y_right.push_back(y[i]);
        }
    }

    node->featureIndex = bestFeature;
    node->threshold = bestThreshold;
    node->left = buildTree(X_left, y_left, depth + 1, maxDepth, minSamplesSplit);
    node->right = buildTree(X_right, y_right, depth + 1, maxDepth, minSamplesSplit);
    return node;
}

int DecTree::predict(Tree* node, const std::vector<double>& x) {
    if (node->isLeaf)
        return node->label;

    if (x[node->featureIndex] < node->threshold)
        return predict(node->left, x);
    else
        return predict(node->right, x);
}
