#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cuda.h>

//#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* Constant declaration */
#define MAX_BRIGHTNESS  255 /* Maximum gray level */
#define GRAYLEVEL       700 /* No. of gray levels */
#define MAX_FILENAME    256 /* Filename length limit */
#define MAX_BUFFERSIZE  256

/* Global constant declaration */
/* Image storage arrays */

unsigned char* image1_y;
unsigned char* image2_y;

/* struct used to hold image dimensions*/
typedef struct dim
{
  int x_size1;
  int y_size1;
  int x_size2;
  int y_size2;
} dimension;

/*prewitt cuda kernel function*/                                                                                                       
__global__ 
void prewitt(dimension size_gpu, unsigned char * image1_in_d, unsigned char* image2_out_d, double max, double min)
{
  int x_coord = threadIdx.x + blockIdx.x * blockDim.x;
  int y_coord = blockIdx.y;


  /* removes any unnecessary threads */
  if(x_coord > size_gpu.x_size1 || y_coord > size_gpu.y_size1)
  {
    return;
  } 

  int i, j;

  __syncthreads();

  extern __shared__ unsigned char block_image[];
  
  __syncthreads();
  


//initialize shared memory for each block
//each shared memory segment is size [3][1026] to access the surrounding pixels used in the block

  if(x_coord % 1024 == 0)
  {
    for(i = -1; i < 2; i++)
    {
      for(j = -1; j < 1025; j++)
      {
        if(j + blockIdx.x * blockDim.x == size_gpu.x_size1 + 1)
        {
          break;
        }
        
        if( ((int)(j + blockIdx.x * blockDim.x) < 0) || (y_coord + i < 0) || (y_coord + i >= size_gpu.y_size1) || (j + blockIdx.x * blockDim.x >= size_gpu.x_size1) )
        {
          block_image[(i+1) * 1026 + (j+1)] = NULL;
        }
        else
        {
          block_image[(i+1) * 1026 + (j+1)] = image1_in_d[ (blockIdx.x * blockDim.x) + j + ((y_coord + i) * size_gpu.x_size1) ];
        }
      }
    } 
  }

  __syncthreads();

  double grad; 
  int kernel[3][3] = 
	{	{-1, 0, 1},
		{-1, 0, 1},
		{-1, 0, 1}
 	};  


//normalizes the pixel in the grayscale image and inputs the new pixel into image2_out_d

  if(x_coord != 0 && x_coord != size_gpu.x_size1 - 1 && y_coord != 0 && y_coord != size_gpu.y_size1 - 1)
  {
    grad = 0.0;
    for(j = -1; j <= 1; j++)
    {
      for(i = -1; i <= 1; i++)
      {
        if(threadIdx.x != 0 && block_image[(j + 1) * 1026 + threadIdx.x + i] != NULL)
        {
          grad += kernel[j + 1][i + 1] * block_image[ (j + 1) * 1026 + threadIdx.x + i];
        }
        else if(block_image[(j + 1) * 1026 + threadIdx.x + 1 + i] != NULL)
	{
          grad += kernel[j + 1][i + 1] * block_image[ (j + 1) * 1026 + threadIdx.x + 1 + i];
	}
      }
    }

    grad = 255 * (grad - min) / (max - min);
    image2_out_d[y_coord * size_gpu.x_size1 + x_coord] = grad;
  }

  __syncthreads();
}





int main(void)
{
///////Image input from mypgm.h
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
  dimension size;
  char file_name[MAX_FILENAME];
  char buffer[MAX_BUFFERSIZE];
  FILE *fp; /* File pointer */
  int max_gray; /* Maximum gray level */
  int x, y; /* Loop variable */

  /* Input file open */
  printf("\n-----------------------------------------------------\n");
  printf("Monochromatic image file input routine \n");
  printf("-----------------------------------------------------\n\n");
  printf("     Only pgm binary file is acceptable\n\n");
  printf("Name of input image file? (*.pgm) : ");
  scanf("%s", file_name);
  fp = fopen(file_name, "rb");
  if (NULL == fp) {
    printf("     The file doesn't exist!\n\n");
    exit(1);
  }

  /* Check of file-type ---P5 */
  fgets(buffer, MAX_BUFFERSIZE, fp);
  if (buffer[0] != 'P' || buffer[1] != '5') {
    printf("     Mistaken file format, not P5!\n\n");
    exit(1);
  }

  /* input of x_size1, y_size1 */
  size.x_size1 = 0;
  size.y_size1 = 0;
  while (size.x_size1 == 0 || size.y_size1 == 0) {
    fgets(buffer, MAX_BUFFERSIZE, fp);
    if (buffer[0] != '#') {
      sscanf(buffer, "%d %d", &size.x_size1, &size.y_size1);
    }
  }

  image1_y = (unsigned char*)malloc(size.x_size1 * size.y_size1 * sizeof(unsigned char));
  image2_y = (unsigned char*)malloc(size.x_size1 * size.y_size1 * sizeof(unsigned char));

  /* input of max_gray */
  max_gray = 0;
  while (max_gray == 0) {
    fgets(buffer, MAX_BUFFERSIZE, fp);
    if (buffer[0] != '#') {
      sscanf(buffer, "%d", &max_gray);
    }
  }

  /* Display of parameters */
  printf("\n     Image width = %d, Image height = %d\n", size.x_size1, size.y_size1);
  printf("     Maximum gray level = %d\n\n",max_gray);
  if (max_gray != MAX_BRIGHTNESS) {
    printf("     Invalid value of maximum gray level!\n\n");
    exit(1);
  }

  /* Input of image data*/
  printf("Total Size: %d\n\n", size.y_size1 * size.x_size1);
  for (y = 0; y < size.y_size1; y++)
  {
    for (x = 0; x < size.x_size1; x++)
    {
      image1_y[y * size.x_size1 + x] = (unsigned char)fgetc(fp);
      image2_y[y * size.x_size1 + x] = 0;
    }
  }

  printf("-----Image data input OK-----\n\n");
  printf("-----------------------------------------------------\n\n");
  
  printf("\n");
  printf("\n");
  printf("\n");
 


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
  
  unsigned char* image1_in_d;
  unsigned char* image2_out_d;
  unsigned char* result;

  const int sizeE = size.y_size1 * size.x_size1 * sizeof(unsigned char);

  //finds the darkest and lightest pixels in the grayscale image
  double max = -DBL_MAX;
  double min = DBL_MAX;
  int i, j;

  int kernel[3][3] = 
  	{	{-1, 0, -1},
		{-1, 0, -1}, 
		{-1, 0, -1}
	};
  double grad;

  for (y = 1; y < size.y_size1 - 1; y++) {
      
    for (x = 1; x < size.x_size1 - 1; x++) {
        
      grad = 0.0;
      for (j = -1; j <= 1; j++) {
          
	    for (i = -1; i <= 1; i++) {
            
	      grad += kernel[j + 1][i + 1] * image1_y[(y + j) * size.x_size1 + x + i];
            
	    }
          
      }
      if (grad < min) min = grad;
      if (grad > max) max = grad;
    }
  }
  
  //if the darkest and lightest pixel is the same then we know the image is blank
  if((int)(max-min) == 0)
  {
    printf("Nothing Exists!!!\n");
    exit(1);
  }

  result = (unsigned char *)malloc(sizeE);

  int block = 1024;
  dim3 grid ((size.x_size1 / 1024) + 1, size.y_size1);

  //allocates the memory in the cuda kernel to hold the imput image and output image
  cudaMalloc((void**)&image1_in_d, sizeE);
  cudaMalloc((void**)&image2_out_d, sizeE);
  
  //copies the image memory into the cuda kernel
  cudaMemcpy(image1_in_d, image1_y , sizeE, cudaMemcpyHostToDevice);
  cudaMemcpy(image2_out_d, image2_y , sizeE, cudaMemcpyHostToDevice);




//starts timer for cuda operation
cudaEvent_t m_start, m_stop;
float m_time;
cudaEventCreate( &m_start );
cudaEventCreate( &m_stop );
cudaEventRecord( m_start, 0 );


    //calls the cuda kernel
    prewitt<<<grid, block, 3 * 1026 * sizeof(unsigned char)>>>(size, image1_in_d, image2_out_d, max, min);


//enda timer for cuda operation and outputs the result
cudaDeviceSynchronize();
cudaEventRecord( m_stop, 0 );
cudaEventSynchronize( m_stop );
cudaEventElapsedTime( &m_time, m_start, m_stop);
printf( "******** Total Running Time of Kernal = %0.5f sec ********\n ", m_time/1000 );
cudaEventDestroy( m_start);
cudaEventDestroy( m_stop);




  //waits for all threads to exit the cuda before continuing
  cudaThreadSynchronize();


  //sets image 2 size to image 1 size
  size.x_size2 = size.x_size1;
  size.y_size2 = size.y_size1;

  //copies over the output image to the results array
  cudaMemcpy(result,image2_out_d, sizeE, cudaMemcpyDeviceToHost);

///////Image output from mypgm.h
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
  
  /* Output file open */
  printf("-----------------------------------------------------\n");
  printf("Monochromatic image file output routine\n");
  printf("-----------------------------------------------------\n\n");
  printf("Name of output image file? (*.pgm) : ");
  scanf("%s",file_name);
  fp = fopen(file_name, "wb");

  /* output of pgm file header information */
  fputs("P5\n", fp);
  fputs("# Created by Image Processing\n", fp);
  fprintf(fp, "%d %d\n", size.x_size1, size.y_size1);
  fprintf(fp, "%d\n", MAX_BRIGHTNESS);

  /* Output of image data */
  for (y = 0; y < size.y_size1; y++) {
    for (x = 0; x < size.x_size1; x++) {
      fputc(result[y * size.x_size1 + x], fp);
    }
  }
  printf("\n-----Image data output OK-----\n\n");
  printf("-----------------------------------------------------\n\n");
  fclose(fp);


////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

  //frees all allocated memory before exiting program
  cudaFree(image2_out_d);
  cudaFree(image1_in_d);
 
  free(result);
  free(image1_y);
  free(image2_y);

	return 0;
}


