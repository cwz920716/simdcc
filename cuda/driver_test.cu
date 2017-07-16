<<<<<<< HEAD
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char **argv) {
=======
#include <stdio.h> 

int main() {
>>>>>>> 9245312cc3e5c713b5bd2bf23eacdeec5bfe67b0
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }
<<<<<<< HEAD

  if (argc != 2) {
      printf("usage: driver_test <Image_Path>\n");
      return -1;
  }

  Mat image;
  image = imread( argv[1], 1 );
  if (!image.data) {
    printf("No image data \n");
    return -1;
  }
  namedWindow("Display Image", WINDOW_AUTOSIZE );
  imshow("Display Image", image);
  waitKey(0);
  return 0;
=======
>>>>>>> 9245312cc3e5c713b5bd2bf23eacdeec5bfe67b0
}
