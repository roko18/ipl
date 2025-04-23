#include <iostream>
#include <omp.h>
#include <cmath>
using namespace std;

float sigmoid(float z) {
    return 1.0f / (1.0f + expf(-z));
}

int main() {
    const int n = 1024;
    float x[n], y[n];

    // Generate data: y = 1 if x > 0.5 else 0
    for (int i = 0; i < n; ++i) {
        x[i] = (i + 1) / 1024.0f;
        y[i] = (x[i] > 0.5f) ? 1.0f : 0.0f;
    }

    float w = 0.0f, b = 0.0f, lr = 0.1f;
    float dw, db;

    for (int epoch = 0; epoch < 1000; ++epoch) {
        dw = 0.0f;
        db = 0.0f;

        // Parallel gradient calculation using OpenMP
        #pragma omp parallel for reduction(+:dw, db)
        for (int i = 0; i < n; ++i) {
            float z = w * x[i] + b;
            float y_pred = sigmoid(z);
            float error = y_pred - y[i];
            dw += x[i] * error / n;
            db += error / n;
        }

        w -= lr * dw;
        b -= lr * db;
    }

    cout << "Learned w: " << w << ", b: " << b << endl;

    // Optional: Test on new data
    float test = 0.4f;
    float pred = sigmoid(w * test + b);
    cout << "Prediction for x = " << test << " is " << pred << endl;

    return 0;
}
