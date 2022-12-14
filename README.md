# Image-Compression
### An Implementation of the Daubechies Wavelets Algorithm using hybrid MPI/OpenMP
For compression ratio 16:1, structural similarity score is 0.94. For compression ratio 64:1, structural similarity score is 0.90.<br />
Benchmark results on different number of nodes and threads showed that with 4 processors and 4 threads per processors, the computing time is optimal. <br />
(*daube4.c: input and outputs are PGM images. The jpg images were converted by OpenCV to display on Github.*)

![Image](lenna1.jpg)
![Image](lenna16.jpg)
![Image](lenna64.jpg)
