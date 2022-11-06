#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 100000000

using namespace std::chrono;

void soma_vetores(float *out, float *a, float *b, int n) {
    for (int i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}

int main() {
    float *a, *b, *out; // vetores na memória RAM

    // Aloca os vetores no host (memória RAM):
    a   = (float*) malloc(sizeof(float) * N);
    b   = (float*) malloc(sizeof(float) * N);
    out = (float*) malloc(sizeof(float) * N);

    // Inicializa os vetores:
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    //############ COMPUTAÇÃO ############
    auto comeco = std::chrono::high_resolution_clock::now();
    soma_vetores(out, a, b, N); // executa a soma
    auto fim = std::chrono::high_resolution_clock::now();

    printf("out[0] = %.3f\n", out[0]);
    double tempo_gasto = duration_cast<milliseconds>(fim - comeco).count();
    printf("Tempo gasto: %.2lfms\n", tempo_gasto);

    // Libera os vetores:
    free(a); 
    free(b);
    free(out);

    return 0;
}
