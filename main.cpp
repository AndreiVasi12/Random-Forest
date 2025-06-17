#include <iostream>
#include "randomforest.h"
#include "csvreader.h"


int main(){
    std::vector<std::vector<double>> X;
    std::vector<int> y;

    readCSV("spam.csv", X, y);
    std::cout << "Loaded " << X.size() << " samples with " << X[0].size() << " features.\n";

    RandomForest forest(10, 5, 2);
    forest.train(X, y);

    for (int i = 0; i < 5; ++i) {
        int pred = forest.predict(X[i]);
        std::cout << "Predicted: " << pred << " | Actual: " << y[i] << "\n";
    }

    return 0;
    

}