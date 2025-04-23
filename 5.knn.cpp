#include <iostream>
#include <cmath>
#include <limits>
#include <omp.h>
using namespace std;

int main() {
    const int n = 1000;
    float train_x[n], distances[n];
    int train_y[n];

    // Generate data: y = 0 if x < 0.5 else 1
    for (int i = 0; i < n; i++) {
        train_x[i] = i / 1000.0f;
        train_y[i] = (train_x[i] < 0.5f) ? 0 : 1;
    }

    float test_x = 0.2f;

    // Compute squared distances in parallel
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float d = train_x[i] - test_x;
        distances[i] = d * d;
    }

    // Find k smallest distances (indices)
    const int k = 10;
    int idx[k];
    for (int i = 0; i < k; i++) idx[i] = -1;

    for (int t = 0; t < k; t++) {
        float minVal = INFINITY;
        int minIdx = -1;
        for (int i = 0; i < n; i++) {
            bool used = false;
            for (int j = 0; j < t; j++)
                if (idx[j] == i) used = true;
            if (!used && distances[i] < minVal) {
                minVal = distances[i];
                minIdx = i;
            }
        }
        idx[t] = minIdx;
    }

    cout << "Nearest neighbors (x, distance, class):" << endl;
    for (int i = 0; i < k; i++) {
        int index = idx[i];
        cout << "x: " << train_x[index]
             << ", Distance: " << distances[index]
             << ", Class: " << train_y[index] << endl;
    }

    // Majority voting
    int count0 = 0, count1 = 0;
    for (int i = 0; i < k; i++) {
        if (train_y[idx[i]] == 0)
            count0++;
        else
            count1++;
    }

    cout << "Predicted class: " << ((count1 > count0) ? 1 : 0) << endl;

    return 0;
}
