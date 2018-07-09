//============================================================================
// Name        : hpc_final_projectSM.cu
// Author      : Rashi Goyal
// Copyright   : Your copyright notice
// Description : Color to Grayscale using CUDA & C++,
// To Run      : nvcc hpc_final_projectSM.cu -lcublas -o hpc_final_projectSM.out
// Note        : Please see report to understand how to run the code to get
//               different outputs
//============================================================================

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
// #include "main.h"

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_WIDTH 8

// kernel implementation for Matrix Multiplication Naive (Non Shared)
__global__ void Convert_to_Grey_2d( int *d_gpu_matrix_in , int *d_gpu_matrix_out , const int WIDTH ){
	
	// 	calculate row & col values for current thread
	unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
	unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

	// 	calculate index using naive logic
	int index=col*WIDTH + row;
	int rgbvalue=d_gpu_matrix_in[index];
	
	// 	extract RGB values from the Pixel data matrix
	int blue 	= rgbvalue % 1000;
	int green 	= ((rgbvalue % 1000000)-blue)/1000;
	int red 	= ((rgbvalue / 1000000)-1000);
	
	// 	calculate grey scale value from RGB values
	d_gpu_matrix_out[index]=(int)((red*.299f) + (green*.587f) + (blue*.114f));

}

// kernel implementation for Matrix Multiplication Naive (Shared)
__global__ void Convert_to_Grey_shared( int *d_gpu_matrix_in , int *d_gpu_matrix_out , const int WIDTH ){
    
    
    __shared__ int rgbvalue;
    // 	calculate row & col values for current thread
    unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
    unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;
    
    // 	calculate index using naive logic
    int index=col*WIDTH + row;
    rgbvalue=d_gpu_matrix_in[index];
    
    // 	extract RGB values from the Pixel data matrix
    int blue 	= rgbvalue % 1000;
    int green 	= ((rgbvalue % 1000000)-blue)/1000;
    int red 	= ((rgbvalue / 1000000)-1000);
    
    // 	calculate grey scale value from RGB values
    d_gpu_matrix_out[index]=(int)((red*.299f) + (green*.587f) + (blue*.114f));
    
}


// 	structure to write data into file
struct lwrite
{
	unsigned long value;
	unsigned      size;
	lwrite( unsigned long value, unsigned size ):
	  value( value ), size( size )
	  { }
};

// 	method to define operator for file operations
inline std::ostream& operator << ( std::ostream& outs, const lwrite& v )
{
	unsigned long value = v.value;
	for (unsigned cntr = 0; cntr < v.size; cntr++, value >>= 8)
	  outs.put( static_cast <char> (value & 0xFF) );
	return outs;
}

// 	method to read data from fstream
template <typename Type>
void read(std::ifstream &fp, Type &result, std::size_t size) {
	fp.read(reinterpret_cast<char*>(&result), size);
}

// 	Bitmap structure to store Bitmap Metadata
struct BMP
{
	typedef int FXPT2DOT30;

	typedef struct {
		FXPT2DOT30 ciexyzX;
		FXPT2DOT30 ciexyzY;
		FXPT2DOT30 ciexyzZ;
	} CIEXYZ;

	typedef struct {
		CIEXYZ  ciexyzRed; 
		CIEXYZ  ciexyzGreen; 
		CIEXYZ  ciexyzBlue; 
	} CIEXYZTRIPLE;

	// 	structure to store Bitmap Headers Metadata
	struct {
		unsigned short bfType; //type of format
		unsigned int   bfSize; //file size
		unsigned short bfReserved1;
		unsigned short bfReserved2;
		unsigned int   bfOffBits;
	} BITMAPFILEHEADER;

	// 	structure to store Bitmap Info Headers Metadata
	struct {
		unsigned int   biSize; 
		unsigned int   biWidth; //width of image
		unsigned int   biHeight; //height of image
		unsigned short biPlanes; 
		unsigned short biBitCount;
		unsigned int   biCompression; //type of compression done
		unsigned int   biSizeImage; 
		unsigned int   biXPelsPerMeter;
		unsigned int   biYPelsPerMeter;
		unsigned int   biClrUsed;
		unsigned int   biClrImportant;
		unsigned int   biRedMask; //Red value of pixel
		unsigned int   biGreenMask; //Green value of pixel
		unsigned int   biBlueMask; //Blue value of Pixel
		unsigned int   biAlphaMask; //alpha value of pixel
		unsigned int   biCSType;
		CIEXYZTRIPLE   biEndpoints;
		unsigned int   biGammaRed;
		unsigned int   biGammaGreen;
		unsigned int   biGammaBlue;
		unsigned int   biIntent;
		unsigned int   biProfileData;
		unsigned int   biProfileSize;
		unsigned int   biReserved;
	} BITMAPINFOHEADER;
};

// 	structure to store RGBA values
typedef struct {
	unsigned char  rgbBlue;
	unsigned char  rgbGreen;
	unsigned char  rgbRed;
	unsigned char  rgbReserved;
} RGBQUAD;

// 	method to extract RGBA 8 bits from BITMAP Data
unsigned char bitextract(const unsigned int byte, const unsigned int mask) {
	if (mask == 0) {
		return 0;
	}

	int maskBufer = mask, maskPadding = 0;

	while (!(maskBufer & 1)) {
		maskBufer >>= 1;
		maskPadding++;
	}

	return (byte & mask) >> maskPadding;
}

// 	method to process BITMAP Data
void process_bitmap_file()
{

	std::cout<<std::endl<<std::endl<<std::endl<<"################### High Performance Computing Project!! (Colored to Greyscale Conversion) ###################"<<std::endl<<std::endl;

	/* Read Bitmap file */
	char *fileName = (char *)"bmp_image1.bmp";

	/* Validate if file is opening or not */
	std::ifstream fileStream(fileName, std::ifstream::binary);
	if (!fileStream) {
		std::cout << "Error opening file '" << fileName << "'." << std::endl;
	}

	BMP bmp;
	
	/* Read BITMAP file headers */
	read(fileStream, bmp.BITMAPFILEHEADER.bfType, sizeof(bmp.BITMAPFILEHEADER.bfType));
	read(fileStream, bmp.BITMAPFILEHEADER.bfSize, sizeof(bmp.BITMAPFILEHEADER.bfSize));
	read(fileStream, bmp.BITMAPFILEHEADER.bfReserved1, sizeof(bmp.BITMAPFILEHEADER.bfReserved1));
	read(fileStream, bmp.BITMAPFILEHEADER.bfReserved2, sizeof(bmp.BITMAPFILEHEADER.bfReserved2));
	read(fileStream, bmp.BITMAPFILEHEADER.bfOffBits, sizeof(bmp.BITMAPFILEHEADER.bfOffBits));

	/* Validate if image is of BMP format */
	if (bmp.BITMAPFILEHEADER.bfType != 0x4D42) {
		std::cout << "Error: '" << fileName << "' is not BMP file." << std::endl;
	}
	
	/* Read BITMAP info headers */
	read(fileStream, bmp.BITMAPINFOHEADER.biSize, sizeof(bmp.BITMAPINFOHEADER.biSize));

	/* Read if BITMAP headers are greater than 12*/
	if (bmp.BITMAPINFOHEADER.biSize >= 12) {
		read(fileStream, bmp.BITMAPINFOHEADER.biWidth, sizeof(bmp.BITMAPINFOHEADER.biWidth));
		read(fileStream, bmp.BITMAPINFOHEADER.biHeight, sizeof(bmp.BITMAPINFOHEADER.biHeight));
		read(fileStream, bmp.BITMAPINFOHEADER.biPlanes, sizeof(bmp.BITMAPINFOHEADER.biPlanes));
		read(fileStream, bmp.BITMAPINFOHEADER.biBitCount, sizeof(bmp.BITMAPINFOHEADER.biBitCount));
	}

	int colorsCount = bmp.BITMAPINFOHEADER.biBitCount >> 3;
	if (colorsCount < 3) {
		colorsCount = 3;
	}

	int bitsOnColor = bmp.BITMAPINFOHEADER.biBitCount / colorsCount;
	int maskValue = (1 << bitsOnColor) - 1;

	/* Read if BITMAP headers are greater than 40 (Bitmap V1)*/
	if (bmp.BITMAPINFOHEADER.biSize >= 40) {
		read(fileStream, bmp.BITMAPINFOHEADER.biCompression, sizeof(bmp.BITMAPINFOHEADER.biCompression));
		read(fileStream, bmp.BITMAPINFOHEADER.biSizeImage, sizeof(bmp.BITMAPINFOHEADER.biSizeImage));
		read(fileStream, bmp.BITMAPINFOHEADER.biXPelsPerMeter, sizeof(bmp.BITMAPINFOHEADER.biXPelsPerMeter));
		read(fileStream, bmp.BITMAPINFOHEADER.biYPelsPerMeter, sizeof(bmp.BITMAPINFOHEADER.biYPelsPerMeter));
		read(fileStream, bmp.BITMAPINFOHEADER.biClrUsed, sizeof(bmp.BITMAPINFOHEADER.biClrUsed));
		read(fileStream, bmp.BITMAPINFOHEADER.biClrImportant, sizeof(bmp.BITMAPINFOHEADER.biClrImportant));
	}

	bmp.BITMAPINFOHEADER.biRedMask = 0;
	bmp.BITMAPINFOHEADER.biGreenMask = 0;
	bmp.BITMAPINFOHEADER.biBlueMask = 0;

	/* Read if BITMAP headers are greater than 52 (Bitmap V2)*/
	if (bmp.BITMAPINFOHEADER.biSize >= 52) {
		read(fileStream, bmp.BITMAPINFOHEADER.biRedMask, sizeof(bmp.BITMAPINFOHEADER.biRedMask));
		read(fileStream, bmp.BITMAPINFOHEADER.biGreenMask, sizeof(bmp.BITMAPINFOHEADER.biGreenMask));
		read(fileStream, bmp.BITMAPINFOHEADER.biBlueMask, sizeof(bmp.BITMAPINFOHEADER.biBlueMask));
	}

	if (bmp.BITMAPINFOHEADER.biRedMask == 0 || bmp.BITMAPINFOHEADER.biGreenMask == 0 || bmp.BITMAPINFOHEADER.biBlueMask == 0) {
		bmp.BITMAPINFOHEADER.biRedMask = maskValue << (bitsOnColor * 2);
		bmp.BITMAPINFOHEADER.biGreenMask = maskValue << bitsOnColor;
		bmp.BITMAPINFOHEADER.biBlueMask = maskValue;
	}

	/* Read if BITMAP headers are greater than 56  (Bitmap V3)*/
	if (bmp.BITMAPINFOHEADER.biSize >= 56) {
		read(fileStream, bmp.BITMAPINFOHEADER.biAlphaMask, sizeof(bmp.BITMAPINFOHEADER.biAlphaMask));
	} else {
		bmp.BITMAPINFOHEADER.biAlphaMask = maskValue << (bitsOnColor * 3);
	}

	/* Read if BITMAP headers are greater than 108 (Bitmap V4)*/
	if (bmp.BITMAPINFOHEADER.biSize >= 108) {
		read(fileStream, bmp.BITMAPINFOHEADER.biCSType, sizeof(bmp.BITMAPINFOHEADER.biCSType));
		read(fileStream, bmp.BITMAPINFOHEADER.biEndpoints, sizeof(bmp.BITMAPINFOHEADER.biEndpoints));
		read(fileStream, bmp.BITMAPINFOHEADER.biGammaRed, sizeof(bmp.BITMAPINFOHEADER.biGammaRed));
		read(fileStream, bmp.BITMAPINFOHEADER.biGammaGreen, sizeof(bmp.BITMAPINFOHEADER.biGammaGreen));
		read(fileStream, bmp.BITMAPINFOHEADER.biGammaBlue, sizeof(bmp.BITMAPINFOHEADER.biGammaBlue));
	}

	/* Read if BITMAP headers are greater than 108 (Bitmap V5)*/
	if (bmp.BITMAPINFOHEADER.biSize >= 124) {
		read(fileStream, bmp.BITMAPINFOHEADER.biIntent, sizeof(bmp.BITMAPINFOHEADER.biIntent));
		read(fileStream, bmp.BITMAPINFOHEADER.biProfileData, sizeof(bmp.BITMAPINFOHEADER.biProfileData));
		read(fileStream, bmp.BITMAPINFOHEADER.biProfileSize, sizeof(bmp.BITMAPINFOHEADER.biProfileSize));
		read(fileStream, bmp.BITMAPINFOHEADER.biReserved, sizeof(bmp.BITMAPINFOHEADER.biReserved));
	}

	if (bmp.BITMAPINFOHEADER.biSize != 12 && bmp.BITMAPINFOHEADER.biSize != 40 && bmp.BITMAPINFOHEADER.biSize != 52 &&
	    bmp.BITMAPINFOHEADER.biSize != 56 && bmp.BITMAPINFOHEADER.biSize != 108 && bmp.BITMAPINFOHEADER.biSize != 124) {
		std::cout << "Error: Unsupported BMP format." << std::endl;
	}

	if (bmp.BITMAPINFOHEADER.biBitCount != 16 && bmp.BITMAPINFOHEADER.biBitCount != 24 && bmp.BITMAPINFOHEADER.biBitCount != 32) {
		std::cout << "Error: Unsupported BMP bit count." << std::endl;
	}

	if (bmp.BITMAPINFOHEADER.biCompression != 0 && bmp.BITMAPINFOHEADER.biCompression != 3) {
		std::cout << "Error: Unsupported BMP compression." << std::endl;
	}

	/* Setting up rows & columns in Image data*/
	int rows=bmp.BITMAPINFOHEADER.biHeight;
	int columns=bmp.BITMAPINFOHEADER.biWidth;

	std::cout<<"------------- IMAGE DETAILS ------------- "<<std::endl<<std::endl;

	std::cout<<" Image Size (Height,Width)   : ("<<rows<<","<<columns<<")"<<std::endl;
	std::cout<<" Number of Pixels            : "<<rows*columns<<std::endl;
	std::cout<<" Image Format                : BMP"<<std::endl<<std::endl;

	/* Print BITMAP headers*/
	std::cout<<"------------- BIT MAP HEADER ------------- "<<std::endl<<std::endl;

	std::cout<<"bfType             : "<<bmp.BITMAPFILEHEADER.bfType              <<std::endl;
	std::cout<<"bfSize             : "<<bmp.BITMAPFILEHEADER.bfSize              <<std::endl;
	std::cout<<"bfReserved1        : "<<bmp.BITMAPFILEHEADER.bfReserved1         <<std::endl;
	std::cout<<"bfReserved2        : "<<bmp.BITMAPFILEHEADER.bfReserved2         <<std::endl;
	std::cout<<"bfOffBits          : "<<bmp.BITMAPFILEHEADER.bfOffBits           <<std::endl<<std::endl;

	std::cout<<"------------- BIT INFO HEADER ------------- "<<std::endl<<std::endl;

	std::cout<<"biSize             : "<<bmp.BITMAPINFOHEADER.biSize              <<std::endl;
	std::cout<<"biWidth            : "<<bmp.BITMAPINFOHEADER.biWidth             <<std::endl;
	std::cout<<"biHeight           : "<<bmp.BITMAPINFOHEADER.biHeight            <<std::endl;
	std::cout<<"biPlanes           : "<<bmp.BITMAPINFOHEADER.biPlanes            <<std::endl;
	std::cout<<"biBitCount         : "<<bmp.BITMAPINFOHEADER.biBitCount          <<std::endl;
	std::cout<<"biCompression      : "<<bmp.BITMAPINFOHEADER.biCompression       <<std::endl;
	std::cout<<"biSizeImage        : "<<bmp.BITMAPINFOHEADER.biSizeImage         <<std::endl;
	std::cout<<"biXPelsPerMeter    : "<<bmp.BITMAPINFOHEADER.biXPelsPerMeter     <<std::endl;
	std::cout<<"biYPelsPerMeter    : "<<bmp.BITMAPINFOHEADER.biYPelsPerMeter     <<std::endl;
	std::cout<<"biClrUsed          : "<<bmp.BITMAPINFOHEADER.biClrUsed           <<std::endl;
	std::cout<<"biClrImportant     : "<<bmp.BITMAPINFOHEADER.biClrImportant      <<std::endl;
	std::cout<<"biRedMask          : "<<bmp.BITMAPINFOHEADER.biRedMask           <<std::endl;
	std::cout<<"biGreenMask        : "<<bmp.BITMAPINFOHEADER.biGreenMask         <<std::endl;
	std::cout<<"biBlueMask         : "<<bmp.BITMAPINFOHEADER.biBlueMask          <<std::endl;
	std::cout<<"biAlphaMask        : "<<bmp.BITMAPINFOHEADER.biAlphaMask         <<std::endl;
	std::cout<<"biCSType           : "<<bmp.BITMAPINFOHEADER.biCSType            <<std::endl;
// 	std::cout<<"biEndpoints        : "<<bmp.BITMAPINFOHEADER.biEndpoints         <<std::endl;
	std::cout<<"biGammaRed         : "<<bmp.BITMAPINFOHEADER.biGammaRed          <<std::endl;
	std::cout<<"biGammaGreen       : "<<bmp.BITMAPINFOHEADER.biGammaGreen        <<std::endl;
	std::cout<<"biGammaBlue        : "<<bmp.BITMAPINFOHEADER.biGammaBlue         <<std::endl;
	std::cout<<"biIntent           : "<<bmp.BITMAPINFOHEADER.biIntent            <<std::endl;
	std::cout<<"biProfileData      : "<<bmp.BITMAPINFOHEADER.biProfileData       <<std::endl;
	std::cout<<"biProfileSize      : "<<bmp.BITMAPINFOHEADER.biProfileSize       <<std::endl;
	std::cout<<"biReserved         : "<<bmp.BITMAPINFOHEADER.biReserved          <<std::endl;
	int linePadding = ((bmp.BITMAPINFOHEADER.biWidth * (bmp.BITMAPINFOHEADER.biBitCount / 8)) % 4) & 3;
	std::cout<<"linePadding        : "<<linePadding<<std::endl;

	/* Setting up RGBA structure to store Image data*/
	RGBQUAD **rgbInfo = new RGBQUAD*[rows];	
	for (unsigned int i = 0; i < rows; i++) {
		rgbInfo[i] = new RGBQUAD[columns];
	}

	/* Setting up Matrixs(rows * columns) to store RGB values of image data*/
	int in_img_data[rows][columns];		// used to store Original Image Data at host
    int serial_img_data[rows][columns];	// used to store Converted Image Data at host
    int out_img_data[rows][columns];	// used to store Converted Image Data at host
    int *d_gpu_matrix_in; 				// used to store Original Image Data at device
    int *d_gpu_matrix_out;				// used to store Converted Image Data at device
	
	/* create device array cudaMalloc ( (void **)&array_name, sizeofmatrixinbytes) */
    cudaMalloc((void **) &d_gpu_matrix_in , rows*columns*sizeof(int) ) ;
    cudaMalloc((void **) &d_gpu_matrix_out , rows*columns*sizeof(int)) ;
                
	/* variables for performance calculations */
	clock_t start;
    clock_t end;
    
	unsigned int bufer;

	/* Starting to read bitmap file for RGBA values */
	for (unsigned int i = 0; i < bmp.BITMAPINFOHEADER.biHeight; i++) {
		for (unsigned int j = 0; j < bmp.BITMAPINFOHEADER.biWidth; j++) {
			read(fileStream, bufer, bmp.BITMAPINFOHEADER.biBitCount / 8);

			rgbInfo[i][j].rgbRed = bitextract(bufer, bmp.BITMAPINFOHEADER.biRedMask);
			rgbInfo[i][j].rgbGreen = bitextract(bufer, bmp.BITMAPINFOHEADER.biGreenMask);
			rgbInfo[i][j].rgbBlue = bitextract(bufer, bmp.BITMAPINFOHEADER.biBlueMask);
			rgbInfo[i][j].rgbReserved = bitextract(bufer, bmp.BITMAPINFOHEADER.biAlphaMask);
			
			/* storing RGBA values as 1+R+G+B Example R=001 G=019 B=255 will compute RGB Value as rgbValue= 1001019255*/
			int rgbValue= 1000000000;
			rgbValue=rgbValue+(((int)rgbInfo[i][j].rgbRed) *1000000);
			rgbValue=rgbValue+(((int)rgbInfo[i][j].rgbGreen) *1000);
			rgbValue=rgbValue+((int)rgbInfo[i][j].rgbBlue);
			
			/* storing data into input matrix for Kernel*/
			in_img_data[i][j]=rgbValue;
			

		}
		fileStream.seekg(linePadding, std::ios_base::cur);
	}    
    
    start=clock();
    for (unsigned int i = 0; i < bmp.BITMAPINFOHEADER.biHeight; i++) {
		for (unsigned int j = 0; j < bmp.BITMAPINFOHEADER.biWidth; j++) {

			/* Code for serial execution of Grey Scale computation */
			int rgbValue= in_img_data[i][j];
			int blue 	= rgbValue % 1000;
			int green 	= ((rgbValue % 1000000)-blue)/1000;
			int red 	= ((rgbValue / 1000000)-1000);
	
			serial_img_data[i][j] =(int)((red*.299f) + (green*.587f) + (blue*.114f));

		}
	}    
    end=clock();

	/* Measuring Performance */
	double dNumOps =rows*columns;
    double dSeconds = (end-start)/1000.0;
    double gflops = 1.0e-9 * dNumOps/dSeconds;
    
	/* Printing Performance */
	std::cout<<"------------- Serial Implementation Performance ------------- "<<std::endl<<std::endl;
	std::cout<<" Number of Operations         : "<<dNumOps<<std::endl;
	std::cout<<" Total time taken             : "<<dSeconds*1000<<"ms"<<std::endl;
	std::cout<<" GFlop per second             : "<<gflops<<std::endl<<std::endl;

	/* Starting of CUDA execution */
    start=clock();

	/* Define dimGrid & dimBlock */
    dim3 dimGrid ( rows/TILE_WIDTH+1 , columns/TILE_WIDTH+1 ,1 ) ;
    dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;

	/* Transfer memory from Host to Device */
    cudaMemcpy ( d_gpu_matrix_in , in_img_data , rows*columns*sizeof(int) , cudaMemcpyHostToDevice ) ;
    
	/* Displaying Kernel Configurations */
	std::cout<<"------------- Kernel Config ------------- "<<std::endl<<std::endl;
	std::cout<<" Grid (x,y,z)                : ("<<rows/TILE_WIDTH+1 <<","<<columns/TILE_WIDTH+1<<",1)"<<std::endl;
	std::cout<<" Block (x,y,z)               : ("<<TILE_WIDTH <<","<<TILE_WIDTH<<",1)"<<std::endl<<std::endl;
	
	/* Kernel Execution */
    Convert_to_Grey_2d <<<dimGrid,dimBlock>>> ( d_gpu_matrix_in ,d_gpu_matrix_out , columns) ;

	std::cout<<" Kernel running...."<<std::endl<<std::endl;

	/* Transfer memory from Device to Host */
    cudaMemcpy(out_img_data , d_gpu_matrix_out , rows*columns*sizeof(int) ,cudaMemcpyDeviceToHost) ;

	/* End of CUDA execution */
    end=clock();
	
	/* Measuring Performance */
	dNumOps =rows*columns;
    dSeconds = (end-start)/1000.0;
    gflops = 1.0e-9 * dNumOps/dSeconds;
    
	/* Printing Performance */
	std::cout<<"-------------Kernel Performance ------------- "<<std::endl<<std::endl;
	std::cout<<" Number of Operations         : "<<dNumOps<<std::endl;
	std::cout<<" Total time taken             : "<<dSeconds*1000<<"ms"<<std::endl;
	std::cout<<" GFlop per second             : "<<gflops<<std::endl<<std::endl;

    /* Starting of shared kernel CUDA execution */
    start=clock();
     
    /* Displaying Kernel Configurations */
    std::cout<<"-------------Shared Kernel Config ------------- "<<std::endl<<std::endl;
    std::cout<<" Grid (x,y,z)                : ("<<rows/TILE_WIDTH+1 <<","<<columns/TILE_WIDTH+1<<",1)"<<std::endl;
    std::cout<<" Block (x,y,z)               : ("<<TILE_WIDTH <<","<<TILE_WIDTH<<",1)"<<std::endl<<std::endl;
    
    /* Kernel Execution */
    Convert_to_Grey_shared <<<dimGrid,dimBlock>>> ( d_gpu_matrix_in ,d_gpu_matrix_out , columns) ;
    
    std::cout<<"Shared Kernel running...."<<std::endl<<std::endl;
    
    /* Transfer memory from Device to Host */
    cudaMemcpy(out_img_data , d_gpu_matrix_out , rows*columns*sizeof(int) ,cudaMemcpyDeviceToHost) ;
    
    /* End of CUDA execution */
    end=clock();
    
    /* Measuring Performance */
    dNumOps =rows*columns;
    dSeconds = (end-start)/1000.0;
    gflops = 1.0e-9 * dNumOps/dSeconds;
    
    /* Printing Performance */
    std::cout<<"-------------Shared Kernel Performance ------------- "<<std::endl<<std::endl;
    std::cout<<" Number of Operations         : "<<dNumOps<<std::endl;
    std::cout<<" Total time taken             : "<<dSeconds*1000<<"ms"<<std::endl;
    std::cout<<" GFlop per second             : "<<gflops<<std::endl<<std::endl;

    
	/* Start converting to greyscale image in BMP format */
    std::cout<<"Converting to GreyScale image"<<std::endl;

    std::ofstream f( "grey_bmp_file.bmp",std::ios::out | std::ios::trunc | std::ios::binary );

	/* setup variables for BITMAPFILEHEADER */
    unsigned long headers_size    = 14  // sizeof( BITMAPFILEHEADER )
                                  + 40; // sizeof( BITMAPINFOHEADER )
    unsigned long padding_size    = (4 - ((columns * 3) % 4)) % 4;
    unsigned long pixel_data_size = rows * ((columns * 3) + padding_size);

	/* Setup BITMAPFILEHEADER for grey image in BMP format */
    f.put( 'B' ).put( 'M' );                           // bfType
    f << lwrite( headers_size + pixel_data_size, 4 );  // bfSize
    f << lwrite( 0,                              2 );  // bfReserved1
    f << lwrite( 0,                              2 );  // bfReserved2
    f << lwrite( headers_size,                   4 );  // bfOffBits

	/* Setup BITMAPINFOHEADER for grey image in BMP format */
    f << lwrite( 40,                             4 );  // biSize
    f << lwrite( columns,                        4 );  // biWidth
    f << lwrite( rows,                           4 );  // biHeight
    f << lwrite( 1,                              2 );  // biPlanes
    f << lwrite( 24,                             2 );  // biBitCount
    f << lwrite( 0,                              4 );  // biCompression=BI_RGB
    f << lwrite( pixel_data_size,                4 );  // biSizeImage
    f << lwrite( 0,                              4 );  // biXPelsPerMeter
    f << lwrite( 0,                              4 );  // biYPelsPerMeter
    f << lwrite( 0,                              4 );  // biClrUsed
    f << lwrite( 0,                              4 );  // biClrImportant

	/* Writing pixel data of grey BMP image */
    for (unsigned row = 0; row<rows; row++)           // bottom-to-top
      {
      for (unsigned col = 0; col < columns; col++)  // left-to-right
        {
        unsigned char red, green, blue;
		
		red=(unsigned char)out_img_data[row][col];
		green=(unsigned char)out_img_data[row][col];
		blue=(unsigned char)out_img_data[row][col];

        f.put( static_cast <char> (blue)  )
         .put( static_cast <char> (green) )
         .put( static_cast <char> (red)   );
        }

      if (linePadding) f << lwrite( 0, linePadding );
      }
      
      std::cout<<"------------- Processing Completed ------------- "<<std::endl<<std::endl;

}
int main()
{
	process_bitmap_file();
	
}


