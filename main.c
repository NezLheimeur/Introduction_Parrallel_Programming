#include <sys/time.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <immintrin.h>	
#include <malloc.h>
#include <errno.h>

#define NB_THREADS 8	  // default value for nb of threads
#define NB_DATA 10000000  //default value for number of data

/* declaration of the thread function routine*/
void *thread_function(void *threadarg);

/* function that  compute the absolute values contained in the vector x (256 bit size <=> 8 float) it's declared inline to accelerate the process during the compilation by adding directly in the code instead of calling it for each thread*/
extern inline __m256 abs_ps(__m256 x){
	const __m256 sign_mask = _mm256_set1_ps(-0.0f); //-0.f =1 << 31		
	return _mm256_andnot_ps(sign_mask, x);		
	
}

/* a new variable type that contain the different mode of computing the threads */
typedef enum{
	SCALAIRE, //scalar method
	VECTORIEL //vectorial method
	
}mode;

/* a structure that contain the thread data */
struct thread_data{
	unsigned int id;	//thread id
	float *U;		//all the data
	long deb;		//start 
	long fin;		//end
	float s;		//result after calculation
	mode mode_calcul;	// computation method
	
};


/*compute the norm of value passed in U using threads, the mode_calc variable is used to choose the mode of computation */
float normPar(float *U, long n, int nb_threads, mode mode_calc){
	
	long i;
	float s=0;	//will contain the fanil result for the norm 


	struct thread_data th_array[nb_threads]; // array of the different threads data
	//double rs[nb_threads]; // will contain the result computed by each thread
	
	pthread_t thread_ptr[nb_threads]; // will contain the different threads

	// set up the parameters for the different threads
	for(i=0;i<nb_threads;i++){
		th_array[i].id=i;
		th_array[i].U=U;
		th_array[i].deb=i*(n/nb_threads);
		th_array[i].fin=(i+1)*(n/nb_threads);
		th_array[i].s=0;
		th_array[i].mode_calcul=mode_calc;
	
		pthread_create(&thread_ptr[i], NULL, thread_function, (void *) &th_array[i]);
	}
	
	//wait for the ends of each thread	
	for(i=0; i<nb_threads; i++){
		pthread_join(thread_ptr[i], NULL);
	}

	//compute the final sum itterating through the threads
	for(i=0;i<nb_threads;i++){
		s+=th_array[i].s;

	}
	return s;	

}

/*the thread routine */
void *thread_function(void *threadarg){

	struct thread_data *thread_pointer_data;
	thread_pointer_data = (struct thread_data *)threadarg;

	pthread_attr_t attr; // initialise a thread attribute with the default values
	//size_t mystacksize = 10000*sizeof(double);
	size_t mystacksize;
	pthread_attr_getstacksize(&attr, &mystacksize); // get the stack size of the current thread
	printf("Thread's nb:%i stack size = %li bytes \n",thread_pointer_data->id, mystacksize);

	float s;
	unsigned int id;
	long i;
	float *U;
	long deb, fin;
	mode mode_calc;
	
	id = thread_pointer_data->id;
	U = thread_pointer_data->U;
	deb = thread_pointer_data->deb;
	fin = thread_pointer_data->fin;
	mode_calc = thread_pointer_data->mode_calcul;
	
	//compute the norm according to the two different modes
	switch (mode_calc)
	{
		case SCALAIRE:
		
			s=0;
			for(i=deb;i<fin;i++){
				s += sqrt(fabs(*(U+i)));
			}
			break;
		
		case VECTORIEL:
		
			s=0;
			
			__m256 *p=(__m256 *)U;
			__m256 u = _mm256_setr_ps(0,0,0,0,0,0,0,0);

			long i=0;
			for(i=deb/8; i<fin/8; i++)
			{
				u=_mm256_add_ps(u,_mm256_sqrt_ps(abs_ps(p[i])));
			}

			float result[8] __attribute__ ((aligned(8*sizeof(float)))); //initialise an anilgned array of size 32 (8*sizeof(float)) that will store the last vector computed previously that contain 8 float

			_mm256_store_ps(result,u); 
			
			// compute the last value that should be stored in the thread data struct
			for(i=0;i<8;i++)
			{
				s+=result[i];
			}
			break;
		
		default:
			printf("invalid calcul mode");
	}
	//store the result of the thread in the associated thread data struct
	thread_pointer_data->s=s;

	//leave the thread
	pthread_exit(NULL);
}			       

/*compute the scalar norm*/
float norm(float *U, long n){
	
	float result=0.0;
	
	long i=0;
	for(i=0;i<n; i++){
		
		result+=sqrt(fabs(*(U+i)));
	}
	return result;
}
	
/*compute the vectorial norm */
float norm_vectorielle(float *U, int n)
{
	float resultat=0;

	__m256 *p=(__m256 *)U;
	__m256 u = _mm256_setr_ps(0,0,0,0,0,0,0,0); //set a 256bit vector with 8 single presion float(32 bit)

	long i=0;
	for(i=0; i<n/8; i++)
	{
		u=_mm256_add_ps(u,_mm256_sqrt_ps(abs_ps(p[i])));//compute the norm on the vectors
	}

	float rs[8] __attribute__ ((aligned(8*sizeof(float))));

	_mm256_store_ps(rs,u); // store the last vector generated in an aligned array of 8 float 

	float s=0;
	for(i=0;i<8;i++)
	{
		resultat+=rs[i]; //computing the norm value by iterationg through the last array and compute the sum
	}

	return resultat;
}

/* function to get the current time in sec */
double now(){

	struct timeval t;
       	double f_t;
	gettimeofday(&t, NULL);
	f_t=t.tv_usec;
	f_t = f_t/1000000.0;
	f_t += t.tv_sec;
	return f_t;
}


int main(int argc, char * argv[])
{
	long N=NB_DATA; //number of data default value 
	int nb_threads= NB_THREADS;
	//check if we define a second parameter during the execution. it should represent the number of data 
	if(argc==2){
		//printf("Greetings, %s!\n", argv[0]);
	       	char *eptr;
		N=strtol(argv[1], &eptr, 10);
		if(N ==0)
		{
			if(errno == EINVAL)
			{
				printf("Conversion error occured: %d\n", errno);
				exit(0);
			}

			if(errno == ERANGE)
				printf("The value provided was out of the range\n");
		}
		printf("the number of data generated is %ld\n", N);
		printf("the number of threads by default is %d\n", nb_threads);
	}
	else if(argc==3){
		
	       	char *eptr;
		N=strtol(argv[1], &eptr, 10);
		nb_threads=atoi(argv[2]);
		if(N ==0 || nb_threads ==0)
		{
			if(errno == EINVAL)
			{
				printf("Conversion error occured: %d\n", errno);
				exit(0);
			}

			if(errno == ERANGE)
				printf("The value provided was out of the range\n");
		}
		printf("the number of data generated is %ld\n", N);
		printf("the number of threads is %d\n", nb_threads);
	}
	else{
		N=NB_DATA;
		nb_threads= NB_THREADS;
		printf("the number of data generated is by default: %ld\n", N);
		printf("the number of threads by default is %d\n", nb_threads);
	}

	//float *v = malloc(N*sizeof(float));
	
	//generate an array that will store the data it should be aligned by 32 bit (8 float) to allow the vector itterating through it and access it without any error
	float *v = aligned_alloc(8*sizeof(float),N*sizeof(float));

	//generating random data
	for(long i=0;i<N;i++){
		v[i]=(float)rand()/(float)RAND_MAX;
	}
	
	float s=0.0;
	double t;
	
	t= now();
	s=norm(v,N);					// compute the scalar norm
	t=now()-t;
	printf("scalaire: S=%f en %f secondes\n",s,t);

	
	printf("#############################\n");

	double t1;	
	t1= now();
	s=norm_vectorielle(v,N); 			// compute the vectorial norm
	t1=now()-t1;
	printf("vectoriel: S=%f en %f secondes, acceleration=%f\n",s,t1,t/t1);

	
	printf("#############################\n");

	double t2;	
	t2= now();
	s=normPar(v,N, nb_threads, SCALAIRE);		// compute the multithreads norm with scalar operations
	t2=now()-t2;
	printf("threads scalaire: S=%f en %f secondes, acceleration=%f\n",s,t2,t/t2);


	printf("#############################\n");

	double t3;	
	t3= now();
	s=normPar(v,N, nb_threads, VECTORIEL);		// compute the multithreads norm with vectorial operations
	t3=now()-t3;
	printf("threads vectoriel: S=%f en %f secondes, acceleration=%f\n",s,t3,t/t3);
}

