#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>

#define LARGURA_MASCARA 129// número ímpar
#define RAIO 64            // (LARGURA_MASCARA - 1) / 2

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

#define T 32

inline void check_cuda(cudaError_t error, const char *filename, const int line) {
  if (error != cudaSuccess) {
    fprintf(stderr, "Error: %s:%d: %s: %s\n", filename, line,
                 cudaGetErrorName(error), cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define CUDACHECK(cmd) check_cuda(cmd, __FILE__, __LINE__)

typedef struct {
  unsigned char red, green, blue;
} PPMPixel;

typedef struct {
  int x, y;
  PPMPixel *data;
} PPMImage;

static PPMImage *readPPM(const char *filename) {
  char buff[16];
  PPMImage *img;
  FILE *fp;
  int c, rgb_comp_color;
  fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Unable to open file '%s'\n", filename);
    exit(1);
  }

  if (!fgets(buff, sizeof(buff), fp)) {
    perror(filename);
    exit(1);
  }

  if (buff[0] != 'P' || buff[1] != '6') {
    fprintf(stderr, "Invalid image format (must be 'P6')\n");
    exit(1);
  }

  img = (PPMImage *)malloc(sizeof(PPMImage));
  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  c = getc(fp);
  while (c == '#') {
    while (getc(fp) != '\n')
      ;
    c = getc(fp);
  }

  ungetc(c, fp);
  if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
    fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
    exit(1);
  }

  if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
    fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
    exit(1);
  }

  if (rgb_comp_color != RGB_COMPONENT_COLOR) {
    fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
    exit(1);
  }

  while (fgetc(fp) != '\n')
    ;
  img->data = (PPMPixel *)malloc(img->x * img->y * sizeof(PPMPixel));

  if (!img) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
    fprintf(stderr, "Error loading image '%s'\n", filename);
    exit(1);
  }

  fclose(fp);
  return img;
}

void writePPM(PPMImage *img, const char *nome_arquivo_out) {
  FILE *fp = fopen(nome_arquivo_out, "wb");
  fprintf(fp, "P6\n");
  fprintf(fp, "# %s\n", COMMENT);
  fprintf(fp, "%d %d\n", img->x, img->y);
  fprintf(fp, "%d\n", RGB_COMPONENT_COLOR);

  fwrite(img->data, 3 * img->x, img->y, fp);
  fclose(fp);
}

__global__ void smoothing_GPUX(unsigned char *channel, int *channel_out, int N, int M) {
  int lrow = threadIdx.y;
  int col = T * blockIdx.x + threadIdx.x;
  int row = T * blockIdx.y + lrow;
  int lindex = threadIdx.x + RAIO;

  __shared__ unsigned char temp[T][LARGURA_MASCARA - 1 + T];
  temp[lrow][lindex] = 0; // init the shared memory

  if (row >= N || col >= M)
    return;

  int gindex = row * M + col;
  // Copy to shared:
  temp[lrow][lindex] = channel[gindex];
  if (threadIdx.x < RAIO) {
    if (col >= RAIO)
      temp[lrow][lindex - RAIO] = channel[gindex - RAIO];
    else
      temp[lrow][lindex - RAIO] = 0;
  } else if (threadIdx.x >= blockDim.x - RAIO) {
    if (col < M - RAIO)
      temp[lrow][lindex + RAIO] = channel[gindex + RAIO];
    else
      temp[lrow][lindex + RAIO] = 0;
  }

  __syncthreads();

  // Apply the stencil:
  int result = 0;
  for (int off = -RAIO; off <= RAIO; off++)
    result += temp[lrow][lindex + off];

  // Store the result:
  channel_out[gindex] = result;
}

__global__ void smoothing_GPUY(int *channel, unsigned char *channel_out, int N, int M) {
  int lcol = threadIdx.x;
  int col = T * blockIdx.x + lcol;
  int row = T * blockIdx.y + threadIdx.y;
  int lindex = threadIdx.y + RAIO; // Transposed

  __shared__ int temp[T][LARGURA_MASCARA - 1 + T]; // Transposed
  temp[lcol][lindex] = 0; // init the shared memory

  if (row >= N || col >= M)
    return;

  int gindex = row * M + col;
  // Copy to shared:
  temp[lcol][lindex] = channel[gindex];
  if (threadIdx.y < RAIO) {
    if (row >= RAIO)
      temp[lcol][lindex - RAIO] = channel[gindex - RAIO * M];
    else
      temp[lcol][lindex - RAIO] = 0;
  } else if (threadIdx.y >= blockDim.y - RAIO) {
    if (row < N - RAIO)
      temp[lcol][lindex + RAIO] = channel[gindex + RAIO * M];
    else
      temp[lcol][lindex + RAIO] = 0;
  }

  __syncthreads();

  // Apply the stencil:
  int result = 0;
  for (int off = -RAIO; off <= RAIO; off++)
    result += temp[lcol][lindex + off];

  // Store the result:
  channel_out[gindex] = result / (LARGURA_MASCARA * LARGURA_MASCARA);
}

void Smoothing(
  unsigned char *__restrict__ red,
  unsigned char *__restrict__ green,
  unsigned char *__restrict__ blue,
  int x, int y) {
  size_t int_size = x * y * sizeof(int);
  size_t char_size = x * y * sizeof(unsigned char);
  unsigned char *d_red, *d_green, *d_blue;
  int *d_red_aux, *d_green_aux, *d_blue_aux;

  // Create a stream for each channel:
  cudaStream_t s1, s2, s3;
  CUDACHECK(cudaStreamCreate(&s1));
  CUDACHECK(cudaStreamCreate(&s2));
  CUDACHECK(cudaStreamCreate(&s3));

  // Alloc data on device:
  CUDACHECK(cudaMalloc(&d_red, char_size));
  CUDACHECK(cudaMalloc(&d_green, char_size));
  CUDACHECK(cudaMalloc(&d_blue, char_size));
  CUDACHECK(cudaMalloc(&d_red_aux, int_size));
  CUDACHECK(cudaMalloc(&d_green_aux, int_size));
  CUDACHECK(cudaMalloc(&d_blue_aux, int_size));

  dim3 dimGrid(ceil((float) x / T), ceil((float) y / T), 1);
  dim3 dimBlock(T, T, 1);

  // Smoothing_GPUX:
  CUDACHECK(cudaMemcpyAsync(d_red, red, char_size, cudaMemcpyHostToDevice, s1));
  smoothing_GPUX<<<dimGrid, dimBlock, 0, s1>>>(d_red, d_red_aux, y, x);

  CUDACHECK(cudaMemcpyAsync(d_green, green, char_size, cudaMemcpyHostToDevice, s2));
  smoothing_GPUX<<<dimGrid, dimBlock, 0, s2>>>(d_green, d_green_aux, y, x);

  CUDACHECK(cudaMemcpyAsync(d_blue, blue, char_size, cudaMemcpyHostToDevice, s3));
  smoothing_GPUX<<<dimGrid, dimBlock, 0, s3>>>(d_blue, d_blue_aux, y, x);

  // Smoothing_GPUY:
  smoothing_GPUY<<<dimGrid, dimBlock, 0, s1>>>(d_red_aux, d_red, y, x);
  CUDACHECK(cudaMemcpyAsync(red, d_red, char_size, cudaMemcpyDeviceToHost, s1));

  smoothing_GPUY<<<dimGrid, dimBlock, 0, s2>>>(d_green_aux, d_green, y, x);
  CUDACHECK(cudaMemcpyAsync(green, d_green, char_size, cudaMemcpyDeviceToHost, s2));

  smoothing_GPUY<<<dimGrid, dimBlock, 0, s3>>>(d_blue_aux, d_blue, y, x);
  CUDACHECK(cudaMemcpyAsync(blue, d_blue, char_size, cudaMemcpyDeviceToHost, s3));

  CUDACHECK(cudaStreamDestroy(s1));
  CUDACHECK(cudaStreamDestroy(s2));
  CUDACHECK(cudaStreamDestroy(s3));

  CUDACHECK(cudaFree(d_red));
  CUDACHECK(cudaFree(d_green));
  CUDACHECK(cudaFree(d_blue));
  CUDACHECK(cudaFree(d_red_aux));
  CUDACHECK(cudaFree(d_green_aux));
  CUDACHECK(cudaFree(d_blue_aux));
}

inline unsigned char *alloc_vet(int size) {
  return (unsigned char *)malloc(size * sizeof(unsigned char));
}

void image_channels(
  const PPMImage *__restrict__ image,
  unsigned char **__restrict__ red,
  unsigned char **__restrict__ green,
  unsigned char **__restrict__ blue, int size) {
  *red = alloc_vet(size);
  *green = alloc_vet(size);
  *blue = alloc_vet(size);

  for (int i = 0; i < size; i++) {
    (*red)[i] = image->data[i].red;
    (*green)[i] = image->data[i].green;
    (*blue)[i] = image->data[i].blue;
  }
}

void image_from_channels(
  const unsigned char *__restrict__ red,
  const unsigned char *__restrict__ green,
  const unsigned char *__restrict__ blue,
  PPMImage *__restrict__ image, int size) {
  for (int i = 0; i < size; i++) {
    image->data[i].red = red[i];
    image->data[i].green = green[i];
    image->data[i].blue = blue[i];
  }
}

int main() {
  unsigned char *red, *green, *blue;
  const char nome_arquivo[] = "/content/drive/MyDrive/secomp/3.ppm";
  const char nome_arquivo_out[] = "/content/drive/MyDrive/secomp/3_out.ppm";
  double t;

  // Read input file
  PPMImage *image = readPPM(nome_arquivo);
  PPMImage *image_output = readPPM(nome_arquivo);

  int size = image->x * image->y;
  // Call Smoothing Kernel
  image_channels(image, &red, &green, &blue, size); // split channels
  Smoothing(red, green, blue, image->x, image->y);
  CUDACHECK(cudaDeviceSynchronize()); // waits for all the work
  image_from_channels(red, green, blue, image_output, size); // build image
  CUDACHECK(cudaGetLastError()); // checks for errors

  // Write result to stdout
  writePPM(image_output, nome_arquivo_out);

  // Cleanup
  free(image->data);
  free(image);
  free(image_output->data);
  free(image_output);
  free(red);
  free(green);
  free(blue);

  return 0;
}
