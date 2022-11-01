#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 100000000

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
    soma_vetores(out, a, b, N); // executa a soma

    printf("out[0] = %.3f\n", out[0]);

    // Libera os vetores:
    free(a); 
    free(b);
    free(out);

    return 0;
}
