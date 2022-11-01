#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 100000000
#define ERRO_MAX 1e-6

__global__ void soma_vetores(float *out, float *a, float *b, int n) {
    for (int i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}

int main() {
    float *a, *b, *out; // vetores na memória RAM 
    float *d_a, *d_b, *d_out; // vetores na memória da GPU

    // Aloca os vetores no host (memória RAM):
    a   = (float*) malloc(sizeof(float) * N);
    b   = (float*) malloc(sizeof(float) * N);
    out = (float*) malloc(sizeof(float) * N);

    // Inicializa os vetores:
    for (int i = 0; i < N; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Aloca os vetores no device (memória da GPU):
    cudaMalloc((void**) &d_a, sizeof(float) * N);
    cudaMalloc((void**) &d_b, sizeof(float) * N);
    cudaMalloc((void**) &d_out, sizeof(float) * N);

    //###### TRANSFERÊNCIA DE DADOS ######
    // Transfere os dados para a GPU:
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    //############ COMPUTAÇÃO ############
    soma_vetores<<<1, 1>>>(d_out, d_a, d_b, N); // executa a soma
    cudaDeviceSynchronize(); // espera até o kernel finalizar
    
    //###### TRANSFERÊNCIA DE DADOS ######
    // Transfere os dados de volta da GPU:
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Testa a resposta:
    for (int i = 0; i < N; i++)
        assert(fabs(out[i] - a[i] - b[i]) < ERRO_MAX);
    printf("out[0] = %.3f\n", out[0]);
    printf("A soma funcionou!\n");

    // Libera os vetores:
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(a); 
    free(b);
    free(out);

    return 0;
}
