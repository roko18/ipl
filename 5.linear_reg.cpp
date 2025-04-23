#include <iostream>
#include <omp.h>
using namespace std;

int main() {
    const int n = 1024;
    float x[n], y[n];

    // Normalize input to prevent overflow
    // here we are creating our large data of 1024 points
    for (int i = 0; i < n; ++i) {
        x[i] = (i + 1) / 1024.0f;  // Normalized x
        y[i] = 2 * x[i] + 3;       // y = 2x + 3
    }

    float w = 0, b = 0, lr = 0.01;
    float dw, db;

    for (int epoch = 0; epoch < 2000; ++epoch) {
        dw = 0.0f;
        db = 0.0f;

        // Parallel gradient computation
        #pragma omp parallel for reduction(+:dw, db)
        for (int i = 0; i < n; ++i) {
            float y_pred = w * x[i] + b;
            float error = y_pred - y[i];
            dw += 2 * x[i] * error / n;
            db += 2 * error / n;
        }

        w -= lr * dw;
        b -= lr * db;

        if(epoch%100==0) {
            cout << "Epoch " << epoch << ": w = " << w << ", b = " << b << endl;
        }
    }

    cout << "Learned w: " << w << ", b: " << b << endl;
    return 0;
}
