#include <math.h>
#include <stdio.h>

float spherical_kernel(const float *x, const float *y, int d) {
    float x_norm = 0.0f, y_norm = 0.0f, dot = 0.0f;
    for (int i = 0; i < d; ++i) {
        x_norm += x[i] * x[i];
        y_norm += y[i] * y[i];
        dot    += x[i] * y[i];
    }
    x_norm = sqrtf(x_norm);
    y_norm = sqrtf(y_norm);
    if (x_norm == 0.0f || y_norm == 0.0f) return 0.0f; // or handle error
    return dot / (x_norm * y_norm);
}

int main() {
    float x[3] = {1, 2, 3};
    float y[3] = {4, 5, 6};
    printf("Cosine similarity: %f\n", spherical_kernel(x, y, 3));
    return 0;
}
