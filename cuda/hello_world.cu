#include <stdio.h>

//   Atributo	Executa no	 Chamado por	 Nota
// __global__	Device	     Host	         Precisa ser void
// __device__	Device	     Device	         Pode retornar qualquer tipo
// __host__	    Host	     Host	         Opcional

__global__ void hello() {
    printf("Hello World da GPU!\n");
}

int main() {
    hello<<<1, 1>>>(); // <<<blocos, threads por bloco>>>, executa de forma assíncrona!
    printf("Hello World da CPU!\n");
    cudaDeviceSynchronize(); // espera até o kernel finalizar
    return 0;
}
