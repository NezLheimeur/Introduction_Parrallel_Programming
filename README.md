
This Project present a small introduction on how we can parallelize a program on a CPU using AVX and Pthreads.
Here we compute an l1 norm of a vector: S=sqrt(abs(X))

to compile the project use the command: gcc -o <executable_name> main.c -mavx -lm -lpthread

For the execution, you can add arguments representing in order the number of data then the number of threads.for example:
./<executable_name> <nb_data> <nb_threads>
You can also represent only the number of data, in this case the number of threads will be 8 (by default). If you run without additional information, the number of data will be 10,000,000 and the number of threads 8.
