//============================================================================
// Name        : convertToGreyScale.cu
// Author      : Rashi Goyal
// Copyright   : Your copyright notice
// Description : Color to Grayscale using CUDA & C++,
// To Run      : nvcc convertToGreyScale.cu -lcublas -o convertToGreyScale.out
// Note        : Please see report to understand how to run the code to get
//               different outputs
//============================================================================

#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <time.h>

using namespace std;

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

const int WIDTH =255;
#define TILE_WIDTH 8

//kernel implementation for Matrix Multiplication Naive (Non Shared)
__global__ void Convert_to_Grey_2d( int *d_gpu_matrix_in , int *d_gpu_matrix_out , const int WIDTH ){
	
// 	calculate row & col values for current thread
	unsigned int col = TILE_WIDTH*blockIdx.x + threadIdx.x ;
	unsigned int row = TILE_WIDTH*blockIdx.y + threadIdx.y ;

	int index=row*WIDTH + col;
	int rgbvalue=d_gpu_matrix_in[index];
	
	int blue 	= rgbvalue % 1000;
	int green 	= ((rgbvalue % 1000000)-blue)/1000;
	int red 	= ((rgbvalue / 1000000)-1000);
	
	d_gpu_matrix_out[index]=int((red*.299f) + (green*.587f) + (blue*.114f));

}

int main() {

	int in_img_data[WIDTH][WIDTH];	// used to store Original Image Data at host
    int out_img_data[WIDTH][WIDTH];	// used to store Converted Image Data at host
    int *d_gpu_matrix_in; 		// used to store Original Image Data at device
    int *d_gpu_matrix_out;		// used to store Converted Image Data at device

	//variables for performance calculations
	clock_t start;
    clock_t end;
	double dSeconds =0.0;
	double gflops =0.0;
	double dNumOps =0.0;

	cout<<endl<<endl<<endl<<"################### High Performance Computing Project!! (Colored to Greyscale Conversion) ###################"<<endl<<endl;

	cout<<" Image Size (Height,Width)   : "<<WIDTH<<","<<WIDTH<<endl;
	cout<<" Number of Pixels            : "<<WIDTH*WIDTH<<endl;
	cout<<" Image Format                : PPM"<<endl<<endl;

	/* Starting to create a Random Color Image based on matrix size */
	std::ofstream img("picture.ppm");
	
	img << "P3" <<endl;
	img << WIDTH << " " << WIDTH << endl;
	img << "255" << endl;	
		
// 	cout<<" Creating Image with random Pixels colors "<<endl;

	/* Allocating pixel colors randomly */
	for (int y=0;y<WIDTH;y++){
		for(int x=0;x<WIDTH;x++){
			
// 			int index=y*WIDTH + x;
			int red=x % 255;
			int green=y % 255;
			int blue=y * x % 255;
									
			img << red << " " << green << " " << blue << endl;
			
			int rgbValue= 1000000000;
			rgbValue=rgbValue+(red *1000000);
			rgbValue=rgbValue+(green *1000);
			rgbValue=rgbValue+(blue);
			
			in_img_data[y][x]=rgbValue;

		}
	}
	img.close();
// 	cout<<"Colored Image created "<<endl;
	
    start=clock();

    dNumOps = WIDTH * WIDTH * 4;    

	//create device array cudaMalloc ( (void **)&array_name, sizeofmatrixinbytes) ;
    cudaMalloc((void **) &d_gpu_matrix_in , WIDTH*WIDTH*sizeof(int) ) ;
    cudaMalloc((void **) &d_gpu_matrix_out , WIDTH*WIDTH*sizeof(int)) ;

	//copy host array to device array
    cudaMemcpy ( d_gpu_matrix_in , in_img_data , WIDTH*WIDTH*sizeof(int) , cudaMemcpyHostToDevice ) ;
    
    dim3 dimGrid ( WIDTH/TILE_WIDTH+1 , WIDTH/TILE_WIDTH+1 ,1 ) ;
    dim3 dimBlock( TILE_WIDTH, TILE_WIDTH, 1 ) ;
    
    //Calling kernel 1 D
	cout<<" Kernel Config.... : "<<endl;
	cout<<" Grid (x,y,z)                : ("<<WIDTH/TILE_WIDTH+1 <<","<<WIDTH/TILE_WIDTH+1<<",1)"<<endl;
	cout<<" Block (x,y,z)               : ("<<TILE_WIDTH <<","<<TILE_WIDTH<<",1)"<<endl<<endl;
	
    Convert_to_Grey_2d <<<dimGrid,dimBlock>>> ( d_gpu_matrix_in ,d_gpu_matrix_out , WIDTH) ;

	cout<<" Kernel running.... : "<<endl<<endl;
    cudaMemcpy(out_img_data , d_gpu_matrix_out , WIDTH*WIDTH*sizeof(int) ,cudaMemcpyDeviceToHost) ;
    end=clock();
	
    //Measuring Performance
    dSeconds = (end-start)/1000.0;
    gflops = 1.0e-9 * dNumOps/dSeconds;
    
	cout<<" Number of Operations         : "<<dNumOps<<endl;
	cout<<" Total time taken             : "<<dSeconds*1000<<endl;
	cout<<" GFlop per second             : "<<gflops<<endl<<endl;

	/* Create GreyScale Image */
	std::ofstream new_img("new_picture.ppm");
		
	new_img << "P3" <<endl;
	new_img << WIDTH << " " << WIDTH << endl;
	new_img << "255" << endl;
	
	
	for (int y=0;y<WIDTH;y++){
		for(int x=0;x<WIDTH;x++){
		
// 			cout<<"("<<y<<","<<x<<") -> "<<out_img_data[y][x]<<endl;				
			new_img << out_img_data[y][x] << " " << out_img_data[y][x] << " " << out_img_data[y][x] << endl;
		}
	}
	new_img.close();
	cout<<endl<<endl<<endl<<" ################## Execution Completed  ################## "<<endl;

}
