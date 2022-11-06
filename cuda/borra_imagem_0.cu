#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define LARGURA_MASCARA 3// número ímpar
#define RAIO 1           // (LARGURA_MASCARA - 1) / 2

#define COMENTARIO "Imagem_Borrada"
#define RGB_MAX 255

using namespace std::chrono;

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

void escrevePPM(PPMImagem *img, const char *nome_arquivo_out) {
    FILE *fp = fopen(nome_arquivo_out, "wb");
    fprintf(fp, "P6\n");
    fprintf(fp, "# %s\n", COMENTARIO);
    fprintf(fp, "%d %d\n", img->x, img->y);
    fprintf(fp, "%d\n", RGB_MAX);

    fwrite(img->imagem, 3 * img->x, img->y, fp);
    fclose(fp);
}

unsigned char *aloca_vet(int tam) {
    return (unsigned char *) malloc(tam * sizeof(unsigned char));
}

void borra_imagem(
        unsigned char *vermelho,
        unsigned char *verde,
        unsigned char *azul,
        int x, int y) {
    int tam = x * y;
    int *vermelho_aux = (int *) malloc(tam * sizeof(int)),
        *verde_aux = (int *) malloc(tam * sizeof(int)),
        *azul_aux = (int *) malloc(tam * sizeof(int));
    unsigned char *vermelho_out = aloca_vet(tam),
        *verde_out = aloca_vet(tam),
        *azul_out = aloca_vet(tam);

    for (int i = 0; i < y; i++)
        for (int j = 0; j < x; j++) {
            vermelho_aux[i * x + j] = verde_aux[i * x + j] = azul_aux[i * x + j] = 0;
            for (int i_offset = i - RAIO; i_offset <= i + RAIO; i_offset++)
                for (int j_offset = j - RAIO; j_offset <= j + RAIO; j_offset++) {
                    if (0 <= i && i < y && 0 <= j && j < x) {
                      vermelho_aux[i * x + j] += vermelho[i_offset * x + j_offset];
                      verde_aux[i * x + j] += verde[i_offset * x + j_offset];
                      azul_aux[i * x + j] += azul[i_offset * x + j_offset];
                    }
                }
            vermelho_out[i * x + j] =  vermelho_aux[i * x + j] / (LARGURA_MASCARA * LARGURA_MASCARA);
            verde_out[i * x + j] =  verde_aux[i * x + j] / (LARGURA_MASCARA * LARGURA_MASCARA);
            azul_out[i * x + j] =  azul_aux[i * x + j] / (LARGURA_MASCARA * LARGURA_MASCARA);
        }

    memcpy(vermelho, vermelho_out, tam * sizeof(unsigned char));
    memcpy(verde, verde_out, tam * sizeof(unsigned char));
    memcpy(azul, azul_out, tam * sizeof(unsigned char));

    free(vermelho_aux);
    free(verde_aux);
    free(azul_aux);
    free(vermelho_out);
    free(verde_out);
    free(azul_out);
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
    const char nome_arquivo[] = "/content/drive/MyDrive/secomp/4.ppm";
    const char nome_arquivo_out[] = "/content/drive/MyDrive/secomp/4_out.ppm";

    // Lê a imagem de entrada:
    PPMImagem *img = lePPM(nome_arquivo, 0);
    PPMImagem *img_output = lePPM(nome_arquivo, 1);
    int tam = img->x * img->y;

    auto comeco = std::chrono::high_resolution_clock::now();

    canais_da_imagem(img, &vermelho, &verde, &azul, tam);
    borra_imagem(vermelho, verde, azul, img->x, img->y);
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
