#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#define LARGURA_MASCARA 31// número ímpar
#define RAIO 15           // (LARGURA_MASCARA - 1) / 2

#define COMENTARIO "Imagem_Borrada"
#define RGB_MAX 255

#define LADO_BLOCO 32// os blocos 2D terão LADO_BLOCOxLADO_BLOCO threads

using namespace std::chrono;

void checacuda(const cudaError_t erro, const char *nome_arquivo, const int linha) {
    if (erro != cudaSuccess) {
        fprintf(stdout, "Erro: %s:%d: %s: %s\n", nome_arquivo, linha,
                cudaGetErrorName(erro), cudaGetErrorString(erro));
        exit(EXIT_FAILURE);
    }
}

#define CHECA_CUDA(cmd) checacuda(cmd, __FILE__, __LINE__)

typedef struct {
    unsigned char vermelho, verde, azul;
} PPMPixel;

typedef struct {
    int x, y;
    PPMPixel *imagem;
} PPMImagem;

PPMImagem *lePPM(const char *nome_arquivo, char pula_pixeis) {
    char buff[16];
    PPMImagem *img;
    FILE *fp;
    int c, rgb_max;
    fp = fopen(nome_arquivo, "rb");
    if (!fp) {
        fprintf(stdout, "Incapaz de abrir o arquivo '%s'\n", nome_arquivo);
        exit(EXIT_FAILURE);
    }

    if (!fgets(buff, sizeof(buff), fp)) {// lê o formato da imagem
        perror(nome_arquivo);
        exit(EXIT_FAILURE);
    }

    if (buff[0] != 'P' || buff[1] != '6') {// valida o formato da imagem
        fprintf(stdout, "Formato de imagem inválido (precisa ser 'P6')\n");
        exit(EXIT_FAILURE);
    }

    c = getc(fp);
    while (c == '#') {// ignora os comentários
        while (getc(fp) != '\n');
        c = getc(fp);
    }

    img = (PPMImagem *) malloc(sizeof(PPMImagem));
    ungetc(c, fp);
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
        fprintf(stdout, "Tamahno inválido (erro ao carregar '%s')\n", nome_arquivo);
        exit(EXIT_FAILURE);
    }

    if (fscanf(fp, "%d", &rgb_max) != 1) {
        fprintf(stdout, "Valor máximo por componente RGB inválido (erro ao carregar '%s')\n", nome_arquivo);
        exit(EXIT_FAILURE);
    }

    if (rgb_max != RGB_MAX) {
        fprintf(stdout, "'%s' não possui componentes de 8-bits\n", nome_arquivo);
        exit(EXIT_FAILURE);
    }

    img->imagem = (PPMPixel *) malloc(img->x * img->y * sizeof(PPMPixel));
    if (!pula_pixeis) {
        while (fgetc(fp) != '\n');

        if (fread(img->imagem, 3 * img->x, img->y, fp) != img->y) {// lê os pixeis
            fprintf(stdout, "Erro ao carregar imagem '%s'\n", nome_arquivo);
            exit(EXIT_FAILURE);
        }
    }

    fclose(fp);
    return img;
}

void escrevePPM(PPMImagem *img, const const char *nome_arquivo_out) {
    FILE *fp = fopen(nome_arquivo_out, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "# %s\n", COMENTARIO);
    fprintf(fp, "%d %d\n", img->x, img->y);
    fprintf(fp, "%d\n", RGB_MAX);

    fwrite(img->imagem, 3 * img->x, img->y, fp);
    fclose(fp);
}

__global__ void borra_X(unsigned char *canal, int *canal_out, int N, int M) {
    int llinha = threadIdx.y;// linha da thread localmente dentro do bloco
    int lcoluna = threadIdx.x + RAIO;// coluna da thread dentro da matriz compartilhada
    // Coordenadas globais da thread:
    int linha = LADO_BLOCO * blockIdx.y + threadIdx.y;
    int coluna = LADO_BLOCO * blockIdx.x + threadIdx.x;

    __shared__ unsigned char temp[LADO_BLOCO][LARGURA_MASCARA - 1 + LADO_BLOCO];
    temp[llinha][lcoluna] = 0;// inicializa a memória compartilhada

    if (linha >= N || coluna >= M)
        return;// threads fora da imagem

    int gindice = linha * M + coluna;// indice global da thread dentro do canal
    // Cópia otimizada de dados para a memória compartilhada (cada thread copia até dois valores):
    temp[llinha][lcoluna] = canal[gindice];//copia o dado correspondente a essa thread
    if (threadIdx.x < RAIO) {// essa thread vai copiar um dado da borda esquerda
        if (coluna >= RAIO)
            temp[llinha][lcoluna - RAIO] = canal[gindice - RAIO];
        else
            temp[llinha][lcoluna - RAIO] = 0;// se a borda sai da imagem coloca zero
    } else if (threadIdx.x >= blockDim.x - RAIO) {// essa thread vai copiar um dado da borda direita
        if (coluna < M - RAIO)
            temp[llinha][lcoluna + RAIO] = canal[gindice + RAIO];
        else
            temp[llinha][lcoluna + RAIO] = 0;// se a borda sai da imagem coloca zero
    }

    __syncthreads();// espera a memória compartilhada ser completamente populada

    // Aplica o stencil:
    int result = 0;
    for (int offset = -RAIO; offset <= RAIO; offset++)
        result += temp[llinha][lcoluna + offset];

    // Salva o resultado final na memória global:
    canal_out[gindice] = result;
}

__global__ void borra_Y(int *canal, unsigned char *canal_out, int N, int M) {
    int llinha = threadIdx.y + RAIO;// linha da thread dentro da matriz compartilhada (transposta)
    int lcol = threadIdx.x;// coluna da thread localmente dentro do bloco
    // Coordenadas globais da thread:
    int linha = LADO_BLOCO * blockIdx.y + threadIdx.y;
    int coluna = LADO_BLOCO * blockIdx.x + threadIdx.x;

    // A matriz compartilhada está transposta:
    __shared__ int temp[LADO_BLOCO][LARGURA_MASCARA - 1 + LADO_BLOCO];
    temp[lcol][llinha] = 0;// inicializa a memória compartilhada

    if (linha >= N || coluna >= M)
        return;// threads fora da imagem

    int gindice = linha * M + coluna;// indice global da thread dentro do canal
    // Cópia otimizada de dados para a memória compartilhada (cada thread copia até dois valores):
    temp[lcol][llinha] = canal[gindice];//copia o dado correspondente a essa thread
    if (threadIdx.y < RAIO) {// essa thread vai copiar um dado da borda superior
        if (linha >= RAIO)
            temp[lcol][llinha - RAIO] = canal[gindice - RAIO * M];
        else
            temp[lcol][llinha - RAIO] = 0;// se a borda sai da imagem coloca zero
    } else if (threadIdx.y >= blockDim.y - RAIO) {// essa thread vai copiar um dado da borda inferior
        if (linha < N - RAIO)
            temp[lcol][llinha + RAIO] = canal[gindice + RAIO * M];
        else
            temp[lcol][llinha + RAIO] = 0;// se a borda sai da imagem coloca zero
    }

    __syncthreads();// espera a memória compartilhada ser completamente populada

    // Aplica o stencil:
    int result = 0;
    for (int offset = -RAIO; offset <= RAIO; offset++)
        result += temp[lcol][llinha + offset];

    // Salva o resultado final na memória global calculando a divisão faltante:
    canal_out[gindice] = result / (LARGURA_MASCARA * LARGURA_MASCARA);
}

void borra_imagem(
        unsigned char *vermelho,
        unsigned char *verde,
        unsigned char *azul,
        int x, int y) {
    // As saídas intermediárias serão armazenadas em vetores de inteiros para evitar overflow:
    size_t tam_char = x * y * sizeof(unsigned char);
    size_t tam_int = x * y * sizeof(int);
    unsigned char *d_vermelho, *d_verde, *d_azul;
    int *d_vermelho_aux, *d_verde_aux, *d_azul_aux;

    // Cria uma Stream para cada canal da imagem:
    cudaStream_t stream_vermelho, stream_verde, stream_azul;
    CHECA_CUDA(cudaStreamCreate(&stream_vermelho));
    CHECA_CUDA(cudaStreamCreate(&stream_verde));
    CHECA_CUDA(cudaStreamCreate(&stream_azul));

    // Aloca os canais na GPU:
    CHECA_CUDA(cudaMalloc(&d_vermelho, tam_char));
    CHECA_CUDA(cudaMalloc(&d_verde, tam_char));
    CHECA_CUDA(cudaMalloc(&d_azul, tam_char));
    CHECA_CUDA(cudaMalloc(&d_vermelho_aux, tam_int));
    CHECA_CUDA(cudaMalloc(&d_verde_aux, tam_int));
    CHECA_CUDA(cudaMalloc(&d_azul_aux, tam_int));

    dim3 dimGrade(ceil((float) x / LADO_BLOCO), ceil((float) y / LADO_BLOCO), 1);
    dim3 dimBloco(LADO_BLOCO, LADO_BLOCO, 1);

    // borra_X:
    CHECA_CUDA(cudaMemcpyAsync(d_vermelho, vermelho, tam_char, cudaMemcpyHostToDevice, stream_vermelho));
    borra_X<<<dimGrade, dimBloco, 0, stream_vermelho>>>(d_vermelho, d_vermelho_aux, y, x);

    CHECA_CUDA(cudaMemcpyAsync(d_verde, verde, tam_char, cudaMemcpyHostToDevice, stream_verde));
    borra_X<<<dimGrade, dimBloco, 0, stream_verde>>>(d_verde, d_verde_aux, y, x);

    CHECA_CUDA(cudaMemcpyAsync(d_azul, azul, tam_char, cudaMemcpyHostToDevice, stream_azul));
    borra_X<<<dimGrade, dimBloco, 0, stream_azul>>>(d_azul, d_azul_aux, y, x);

    // borra_Y:
    borra_Y<<<dimGrade, dimBloco, 0, stream_vermelho>>>(d_vermelho_aux, d_vermelho, y, x);
    CHECA_CUDA(cudaMemcpyAsync(vermelho, d_vermelho, tam_char, cudaMemcpyDeviceToHost, stream_vermelho));

    borra_Y<<<dimGrade, dimBloco, 0, stream_verde>>>(d_verde_aux, d_verde, y, x);
    CHECA_CUDA(cudaMemcpyAsync(verde, d_verde, tam_char, cudaMemcpyDeviceToHost, stream_verde));

    borra_Y<<<dimGrade, dimBloco, 0, stream_azul>>>(d_azul_aux, d_azul, y, x);
    CHECA_CUDA(cudaMemcpyAsync(azul, d_azul, tam_char, cudaMemcpyDeviceToHost, stream_azul));

    CHECA_CUDA(cudaStreamDestroy(stream_vermelho));
    CHECA_CUDA(cudaStreamDestroy(stream_verde));
    CHECA_CUDA(cudaStreamDestroy(stream_azul));

    CHECA_CUDA(cudaFree(d_vermelho));
    CHECA_CUDA(cudaFree(d_verde));
    CHECA_CUDA(cudaFree(d_azul));
    CHECA_CUDA(cudaFree(d_vermelho_aux));
    CHECA_CUDA(cudaFree(d_verde_aux));
    CHECA_CUDA(cudaFree(d_azul_aux));
}

unsigned char *aloca_vet(int tam) {
    return (unsigned char *) malloc(tam * sizeof(unsigned char));
}

void canais_da_imagem(
        const PPMImagem *img,
        unsigned char **vermelho,
        unsigned char **verde,
        unsigned char **azul, int tam) {
    *vermelho = aloca_vet(tam);
    *verde = aloca_vet(tam);
    *azul = aloca_vet(tam);

    for (int i = 0; i < tam; i++) {
        (*vermelho)[i] = img->imagem[i].vermelho;
        (*verde)[i] = img->imagem[i].verde;
        (*azul)[i] = img->imagem[i].azul;
    }
}

void imagem_dos_canais(
        const unsigned char *vermelho,
        const unsigned char *verde,
        const unsigned char *azul,
        PPMImagem *img, int tam) {
    for (int i = 0; i < tam; i++) {
        img->imagem[i].vermelho = vermelho[i];
        img->imagem[i].verde = verde[i];
        img->imagem[i].azul = azul[i];
    }
}

int main() {
    unsigned char *vermelho, *verde, *azul;// canais da imagem de entrada
    const char nome_arquivo[] = "/content/drive/MyDrive/secomp/nome_arquivo.ppm";
    const char nome_arquivo_out[] = "/content/drive/MyDrive/secomp/nome_arquivo_out.ppm";

    // Lê a imagem de entrada:
    PPMImagem *img = lePPM(nome_arquivo, 0);
    PPMImagem *img_output = lePPM(nome_arquivo, 1);
    int tam = img->x * img->y;

    auto comeco = std::chrono::high_resolution_clock::now();

    canais_da_imagem(img, &vermelho, &verde, &azul, tam);
    borra_imagem(vermelho, verde, azul, img->x, img->y);
    CHECA_CUDA(cudaDeviceSynchronize());// espera até os kerneis finalizarem
    imagem_dos_canais(vermelho, verde, azul, img_output, tam);

    auto fim = std::chrono::high_resolution_clock::now();

    escrevePPM(img_output, nome_arquivo_out);
    double tempo_gasto = duration_cast<milliseconds>(fim - comeco).count();
    printf("Tempo gasto: %.2lfms\n", tempo_gasto);

    // Libera os dados alocados:
    free(img->imagem);
    free(img);
    free(img_output->imagem);
    free(img_output);
    free(vermelho);
    free(verde);
    free(azul);

    return 0;
}
