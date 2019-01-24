/*****************************************************
 *
 * S O R algorithm
 * ("Red-Black" solution to LaPlace approximation)
 *
 * Parallel version
 *
 * BY
 * Akhil Tummala & Madhukar Enugurthi
 *
 *****************************************************/

 /* 
 * Compile with:
 * mpicc -o lap lap.c 
 */
 
 /* 
 * Execute with:
 * mpirun -n 4 ./lap 
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <mpi.h>

#define MAX_SIZE 4096
#define EVEN_TURN 0 /* shall we calculate the 'red' or the 'black' elements */
#define ODD_TURN  1

#define MS 1	/* setting a message type */
#define SS 2	/* setting a message type */
#define SM 3	/* setting a message type */

MPI_Status status;

double *A;

int	size = 2048;				/* matrix size		*/
int	maxnum = 15.0;				/* max number of element*/
char *Init = "rand";				/* matrix init type	*/

double difflimit = 0.02048; // difflimtit = 0.00001 * size		/* Stop condition  */

double w = 0.5;					/* relaxation factor	*/
int	PRINT = 0;					/* print switch		*/

/* forward declarations */
int work();
void Init_Matrix();
void Print_Matrix();
int Read_Options(int, char **);

int 
main(int argc, char **argv)
{
    int iter;
	int rank, nproc;
	double start_time, end_time;
    
	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	

	if(nproc == 1)
	{
		Read_Options(argc,argv);	/* Read arguments	*/
		A = malloc((size + 2) * (size + 2) * sizeof(double));
		Init_Matrix();		/* Init the matrix	*/
		
		start_time = MPI_Wtime();
		iter = work_1();
		end_time = MPI_Wtime();
		
		if (PRINT == 1)
		Print_Matrix();
    
		printf("\nNumber of iterations = %d\n", iter);
		printf("\ntime: %f\n\n", end_time - start_time);
		free(A);
	}
	else if(nproc > 1)
	{
		if(rank == 0)
		{
			Read_Options(argc,argv);	/* Read arguments	*/
			A = malloc((size + 2) * (size + 2) * sizeof(double));
			Init_Matrix();		/* Init the matrix	*/
			
		}
		
		// broadcasting the size, difflimit and , relaxation factor 
		MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&difflimit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Bcast(&w, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		
		if(rank == 0)
		{	start_time = MPI_Wtime();
			iter = work_slaves(nproc, rank);
			end_time = MPI_Wtime();
    
			if (PRINT == 1)
			{
				Print_Matrix();
			}
			
			printf("\nNumber of Nodes = %d\n", nproc);
			printf("\nNumber of iterations = %d\n", iter);
			printf("\ntime: %f\n\n", end_time - start_time);

			free(A);
		}
		else
		{
			work_slaves(nproc,rank);
		}
	}
	
	MPI_Finalize();
}

int
work_1()
{
    double prevmax_even, prevmax_odd, maxi, sum;
    int	m, n;
    int finished = 0;
    int turn = EVEN_TURN;
    int iteration = 0;

    prevmax_even = 0.0;
    prevmax_odd = 0.0;
    int cols = size+2;
    
    while (!finished) 
	{
		iteration++;
		if (turn == EVEN_TURN) {
			/* CALCULATE part A - even elements */
			for (m = 1; m < size+1; m++) {
			for (n = 1; n < size+1; n++) {
				if (((m + n) % 2) == 0)
				A[m* cols + n] = (1 - w) * A[m*cols + n] 
					+ w * (A[(m-1) * cols + n] + A[(m+1) * cols + n] 
						+ A[m * cols +(n-1)] + A[m*cols + (n+1)]) / 4;
			}
			}
	    
		/* Calculate the maximum sum of the elements */
	    maxi = -999999.0;
	    for (m = 1; m < size+1; m++) {
		sum = 0.0;
		for (n = 1; n < size+1; n++)
		    sum += A[m*cols + n];
		if (sum > maxi)
		    maxi = sum;
	    }
	    /* Compare the sum with the prev sum, i.e., check wether 
	     * we are finished or not. */
	    if (fabs(maxi - prevmax_even) <= difflimit)
		finished = 1;
	    
		
		prevmax_even = maxi;
	    turn = ODD_TURN;

		} else if (turn == ODD_TURN) {
			/* CALCULATE part B - odd elements*/
			for (m = 1; m < size+1; m++) {
			for (n = 1; n < size+1; n++) {
				if (((m + n) % 2) == 1)
					A[m* cols + n] = (1 - w) * A[m*cols + n] 
					+ w * (A[(m-1) * cols + n] + A[(m+1) * cols + n] 
						+ A[m * cols +(n-1)] + A[m*cols + (n+1)]) / 4;
				}
			}
	    /* Calculate the maximum sum of the elements */
	    maxi = -999999.0;
	    for (m = 1; m < size+1; m++) {
		sum = 0.0;
		for (n = 1; n < size+1; n++)
		    sum += A[m*cols + n];	
		if (sum > maxi)			
		    maxi = sum;
	    }
	    /* Compare the sum with the prev sum, i.e., check wether 
	     * we are finished or not. */
	    if (fabs(maxi - prevmax_odd) <= difflimit)
		finished = 1;
	    
		
		prevmax_odd = maxi;
	    turn = EVEN_TURN;
		} 
		else {
			/* something is very wrong... */
			printf("PANIC: Something is really wrong!!!\n");
			exit(-1);
		}
		
		if (iteration > 100000) {
	    /* exit if we don't converge fast enough */
	    printf("Max number of iterations reached! Exit!\n");
	    finished = 1;
		}
    }
    return iteration;
}

int
work_slaves(int nproc, int rank)
{
	double prevmax_even, prevmax_odd, sum;
    int	m, n;
    int finished = 0;
    int turn = EVEN_TURN;
    int iteration = 0;
	int i,j;

    prevmax_even = 0.0;
    prevmax_odd = 0.0;
	  
    int cols = size+2;
    int offset = size/nproc;
	int rows = offset+2;
	// Creating a dummy matrix for each node
	double *B = malloc(rows * cols * sizeof(double));
	// Disturbuting the Data of Matrix A for each B
	MPI_Scatter(&A[cols], (rows - 2) * cols, MPI_DOUBLE,&B[cols], (rows - 2) * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if(nproc == 2)
	{
		if(rank == 0)
		{
			// copying first and last rows to the first and last node.
			memcpy(&B[0], &A[0], cols * sizeof(double));
			
			MPI_Send(&A[(size +1) * cols ], cols, MPI_DOUBLE,1, MS, MPI_COMM_WORLD);
						
		}
		else
		{
			MPI_Recv(&B[(rows - 1) * cols], cols, MPI_DOUBLE, 0,MS,MPI_COMM_WORLD, &status);
			
		}
	}
	
	else if(nproc == 4)
	{
		if(rank == 0)
		{
			// copying first and last rows to the first and last node.
			memcpy(&B[0], &A[0], cols * sizeof(double));
			MPI_Send(&A[(size +1) * cols ], cols, MPI_DOUBLE,3, MS, MPI_COMM_WORLD);
			
		}
		else if(rank == 3)
		{
			MPI_Recv(&B[(rows - 1) * cols], cols, MPI_DOUBLE, 0,MS,MPI_COMM_WORLD, &status);
		}
		
	}
	
	else if(nproc == 8)
	{
		if(rank == 0)
		{
			// copying first and last rows to the first and last node.
			memcpy(&B[0], &A[0], cols * sizeof(double));
			MPI_Send(&A[(size +1) * cols ], cols, MPI_DOUBLE,7, MS, MPI_COMM_WORLD);
			
		}
		else if(rank == 7)
		{
			MPI_Recv(&B[(rows - 1) * cols], cols, MPI_DOUBLE, 0,MS,MPI_COMM_WORLD, &status);
		}
	}
	
	
    while (!finished) 
	{
		
		if(nproc == 2)
		{
			if(rank == 0)
			{
				// updating the last and first row of the matrix in each node
				MPI_Send(&B[(rows - 2) * cols], cols, MPI_DOUBLE,1, MS, MPI_COMM_WORLD);
				MPI_Recv(&B[(rows - 1) * cols], cols, MPI_DOUBLE, 1, SM,MPI_COMM_WORLD, &status);
			}
			else if(rank == 1)
			{
				// updating the last and first row of the matrix in each node
				MPI_Recv(&B[0], cols, MPI_DOUBLE, 0, MS,MPI_COMM_WORLD, &status);
				MPI_Send(&B[cols], cols, MPI_DOUBLE,0, SM, MPI_COMM_WORLD);
				
			}
			
		}
		
		else if(nproc == 4)
		{
			if(rank == 0)
			{
				// updating the last and first row of the matrix in each node
				MPI_Send(&B[(rows - 2) * cols], cols, MPI_DOUBLE,1, MS, MPI_COMM_WORLD);
				MPI_Recv(&B[(rows - 1) * cols], cols, MPI_DOUBLE, 1, SM,MPI_COMM_WORLD, &status);
			}
			else if(rank == 1)
			{
				// updating the last and first row of the matrix in each node
				MPI_Recv(&B[0], cols, MPI_DOUBLE, 0, MS,MPI_COMM_WORLD, &status);
				MPI_Send(&B[cols], cols, MPI_DOUBLE,0, SM, MPI_COMM_WORLD);
				
				MPI_Send(&B[(rows - 2) * cols], cols, MPI_DOUBLE,2, SS, MPI_COMM_WORLD);
				MPI_Recv(&B[(rows - 1) * cols], cols, MPI_DOUBLE, 2, SS,MPI_COMM_WORLD, &status);
			}
			else if(rank == 2)
			{
				// updating the last and first row of the matrix in each node
				MPI_Recv(&B[0], cols, MPI_DOUBLE, 1, SS,MPI_COMM_WORLD, &status);
				MPI_Send(&B[cols], cols, MPI_DOUBLE,1, SS, MPI_COMM_WORLD);
				
				MPI_Send(&B[(rows - 2) * cols], cols, MPI_DOUBLE,3, SS, MPI_COMM_WORLD);
				MPI_Recv(&B[(rows - 1) * cols], cols, MPI_DOUBLE, 3, SS,MPI_COMM_WORLD, &status);
			}
			else
			{
				// updating the last and first row of the matrix in each node
				MPI_Recv(&B[0], cols, MPI_DOUBLE, 2, SS,MPI_COMM_WORLD, &status);
				MPI_Send(&B[cols], cols, MPI_DOUBLE,2, SS, MPI_COMM_WORLD);
			}
		}
		
		else if(nproc == 8)
		{
			if(rank == 0)
			{
				// updating the last and first row of the matrix in each node
				MPI_Send(&B[(rows - 2) * cols], cols, MPI_DOUBLE,1, MS, MPI_COMM_WORLD);
				MPI_Recv(&B[(rows - 1) * cols], cols, MPI_DOUBLE, 1, SM,MPI_COMM_WORLD, &status);
			
			}
			else if(rank == 1)
			{
				// updating the last and first row of the matrix in each node
				MPI_Recv(&B[0], cols, MPI_DOUBLE, 0, MS,MPI_COMM_WORLD, &status);
				MPI_Send(&B[cols], cols, MPI_DOUBLE,0, SM, MPI_COMM_WORLD);
				
				MPI_Send(&B[(rows - 2) * cols], cols, MPI_DOUBLE,2, SS, MPI_COMM_WORLD);
				MPI_Recv(&B[(rows - 1) * cols], cols, MPI_DOUBLE, 2, SS,MPI_COMM_WORLD, &status);
			
			}
			else if(rank == 2)
			{
				// updating the last and first row of the matrix in each node
				MPI_Recv(&B[0], cols, MPI_DOUBLE, 1, SS,MPI_COMM_WORLD, &status);
				MPI_Send(&B[cols], cols, MPI_DOUBLE,1, SS, MPI_COMM_WORLD);
				
				MPI_Send(&B[(rows - 2) * cols], cols, MPI_DOUBLE,3, SS, MPI_COMM_WORLD);
				MPI_Recv(&B[(rows - 1) * cols], cols, MPI_DOUBLE, 3, SS,MPI_COMM_WORLD, &status);
			
			}
			else if(rank == 3)
			{
				// updating the last and first row of the matrix in each node
				MPI_Recv(&B[0], cols, MPI_DOUBLE, 2, SS,MPI_COMM_WORLD, &status);
				MPI_Send(&B[cols], cols, MPI_DOUBLE,2, SS, MPI_COMM_WORLD);
				
				MPI_Send(&B[(rows - 2) * cols], cols, MPI_DOUBLE,4, SS, MPI_COMM_WORLD);
				MPI_Recv(&B[(rows - 1) * cols], cols, MPI_DOUBLE, 4, SS,MPI_COMM_WORLD, &status);
			
			}
			else if(rank == 4)
			{
				// updating the last and first row of the matrix in each node
				MPI_Recv(&B[0], cols, MPI_DOUBLE, 3, SS,MPI_COMM_WORLD, &status);
				MPI_Send(&B[cols], cols, MPI_DOUBLE,3, SS, MPI_COMM_WORLD);
				
				MPI_Send(&B[(rows - 2) * cols], cols, MPI_DOUBLE,5, SS, MPI_COMM_WORLD);
				MPI_Recv(&B[(rows - 1) * cols], cols, MPI_DOUBLE, 5, SS,MPI_COMM_WORLD, &status);
			
			}
			else if(rank == 5)
			{
				// updating the last and first row of the matrix in each node
				MPI_Recv(&B[0], cols, MPI_DOUBLE, 4, SS,MPI_COMM_WORLD, &status);
				MPI_Send(&B[cols], cols, MPI_DOUBLE,4, SS, MPI_COMM_WORLD);
				
				MPI_Send(&B[(rows - 2) * cols], cols, MPI_DOUBLE,6, SS, MPI_COMM_WORLD);
				MPI_Recv(&B[(rows - 1) * cols], cols, MPI_DOUBLE, 6, SS,MPI_COMM_WORLD, &status);
			
			}
			else if(rank == 6)
			{
				// updating the last and first row of the matrix in each node
				MPI_Recv(&B[0], cols, MPI_DOUBLE, 5, SS,MPI_COMM_WORLD, &status);
				MPI_Send(&B[cols], cols, MPI_DOUBLE,5, SS, MPI_COMM_WORLD);
				
				MPI_Send(&B[(rows - 2) * cols], cols, MPI_DOUBLE,7, SS, MPI_COMM_WORLD);
				MPI_Recv(&B[(rows - 1) * cols], cols, MPI_DOUBLE, 7, SS,MPI_COMM_WORLD, &status);
			
			}
			else
			{
				// updating the last and first row of the matrix in each node
				MPI_Recv(&B[0], cols, MPI_DOUBLE, 6, SS,MPI_COMM_WORLD, &status);
				MPI_Send(&B[cols], cols, MPI_DOUBLE,6, SS, MPI_COMM_WORLD);
				
			}
		}
		
		
		
		if(iteration%2 == 0)
		{
			turn = EVEN_TURN;
		}
		else 
		{
			turn = ODD_TURN;
		}
		
		if (turn == EVEN_TURN) {
			/* CALCULATE part A - even elements */
			for (m = 1; m < offset+1; m++) {
			for (n = 1; n < size+1; n++) {
				if (((m + n) % 2) == 0)
				B[m* cols + n] = (1 - w) * B[m*cols + n] 
					+ w * (B[(m-1) * cols + n] + B[(m+1) * cols + n] 
						+ B[m * cols +(n-1)] + B[m*cols + (n+1)]) / 4;
			}
			}
	    
		/* Calculate the maximum sum of the elements */
	    double work_maxi1 = -999999.0;
	    for (m = 1; m < offset+1; m++) {
		sum = 0.0;
		for (n = 1; n < size+1; n++)
		    sum += B[m*cols + n];
		if (sum > work_maxi1)
		    work_maxi1 = sum;
	    }
		
		double final_maxi1;
		MPI_Allreduce(&work_maxi1, &final_maxi1, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		
	    /* Compare the sum with the prev sum, i.e., check wether 
	     * we are finished or not. */
	    if (fabs(final_maxi1 - prevmax_even) <= difflimit)
			finished = 1;
	    		
		prevmax_even = final_maxi1;
		}
		
		if (turn == ODD_TURN) {
			/* CALCULATE part B - odd elements*/
			for (m = 1; m < offset+1; m++) {
			for (n = 1; n < size+1; n++) {
				if (((m + n) % 2) == 1)
					B[m* cols + n] = (1 - w) * B[m*cols + n] 
					+ w * (B[(m-1) * cols + n] + B[(m+1) * cols + n] 
						+ B[m * cols +(n-1)] + B[m*cols + (n+1)]) / 4;
				}
			}
			/* Calculate the maximum sum of the elements */
			double work_maxi2= -999999.0;
			for (m = 1; m < offset+1; m++) {
			sum = 0.0;
			for (n = 1; n < size+1; n++)
				sum += B[m*cols + n];	
			if (sum > work_maxi2)			
				work_maxi2 = sum;
			}
		
			double final_maxi2;
			MPI_Allreduce(&work_maxi2, &final_maxi2, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		
		
			/* Compare the sum with the prev sum, i.e., check wether 
			* we are finished or not. */
			if (fabs(final_maxi2 - prevmax_odd) <= difflimit)
				finished = 1;
	    
		
			prevmax_odd = final_maxi2;
		} 
		
		iteration++;
		
		if (iteration > 100000) {
	    /* exit if we don't converge fast enough */
	    printf("Max number of iterations reached! Exit!\n");
	    finished = 1;
		}
    }
	
	MPI_Gather(&B[1 * cols + 0], (rows - 2) * cols, MPI_DOUBLE,&A[1 * cols + 0], (rows - 2) * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	free(B);
	return iteration;
}
/*--------------------------------------------------------------*/

void
Init_Matrix()
{
    int i, j, dmmy;
 
    int cols = size + 2;
    printf("\nsize      = %dx%d ",size,size);
    printf("\nmaxnum    = %d \n",maxnum);
    printf("difflimit = %.7lf \n",difflimit);
    printf("Init	  = %s \n",Init);
    printf("w	  = %f \n\n",w);
    printf("Initializing matrix...");
 
    /* Initialize all grid elements, including the boundary */
    for (i = 0; i < size+2; i++) {
	for (j = 0; j < size+2; j++) {
	    A[i*cols + j] = 0.0;
	}
    }
    if (strcmp(Init,"count") == 0) {
	for (i = 1; i < size+1; i++){
	    for (j = 1; j < size+1; j++) {
		A[i*cols + j] = (double)i/2;
	    }
	}
    }
    if (strcmp(Init,"rand") == 0) {
	for (i = 1; i < size+1; i++){
	    for (j = 1; j < size+1; j++) {
		A[i*cols + j] = (rand() % maxnum) + 1.0;
	    }
	}
    }
    if (strcmp(Init,"fast") == 0) {
	for (i = 1; i < size+1; i++){
	    dmmy++;
	    for (j = 1; j < size+1; j++) {
		dmmy++;
		if ((dmmy%2) == 0)
		    A[i*cols + j] = 1.0;
		else
		    A[i*cols + j] = 5.0;
	    }
	}
    }

    /* Set the border to the same values as the outermost rows/columns */
    /* fix the corners */
    A[0*cols] = A[cols + 1];
    A[size+1] = A[cols+size];
    A[(size+1)*cols] = A[size*cols + 1];
    A[(size+1)*cols + size + 1] = A[size*cols + size];
    /* fix the top and bottom rows */
    for (i = 1; i < size+1; i++) {
	A[0*cols + i] = A[1*cols + i];
	A[(size+1)*cols + i] = A[size*cols + i];
    }
    /* fix the left and right columns */
    for (i = 1; i < size+1; i++) {
	A[i*cols + 0] = A[i* cols + 1];
	A[i*cols + size+1] = A[i*cols + size];
    }

    printf("done \n\n");
    if (PRINT == 1)
	Print_Matrix();
}

void
Print_Matrix()
{
    int i, j;
	int cols = size + 2;
	
    for (i=0; i<size+2 ;i++){
	for (j=0; j<size+2 ;j++){
	    printf(" %f",A[i*cols + j]);
	}
	printf("\n");
    }
    printf("\n\n");
}



int
Read_Options(int argc, char **argv)
{
    char    *prog;
 
    prog = *argv;
    while (++argv, --argc > 0)
	if (**argv == '-')
	    switch ( *++*argv ) {
	    case 'n':
		--argc;
		size = atoi(*++argv);
		difflimit = 0.00001*size;
		break;
	    case 'h':
		printf("\nHELP: try sor -u \n\n");
		exit(0);
		break;
	    case 'u':
		printf("\nUsage: sor [-n problemsize]\n");
		printf("           [-d difflimit] 0.1-0.000001 \n");
		printf("           [-D] show default values \n");
		printf("           [-h] help \n");
		printf("           [-I init_type] fast/rand/count \n");
		printf("           [-m maxnum] max random no \n");
		printf("           [-P print_switch] 0/1 \n");
		printf("           [-w relaxation_factor] 1.0-0.1 \n\n");
		exit(0);
		break;
	    case 'D':
		printf("\nDefault:  n         = %d ", size);
		printf("\n          difflimit = 0.0001 ");
		printf("\n          Init      = rand" );
		printf("\n          maxnum    = 5 ");
		printf("\n          w         = 0.5 \n");
		printf("\n          P         = 0 \n\n");
		exit(0);
		break;
	    case 'I':
		--argc;
		Init = *++argv;
		break;
	    case 'm':
		--argc;
		maxnum = atoi(*++argv);
		break;
	    case 'd':
		--argc;
		difflimit = atof(*++argv);
		break;
	    case 'w':
		--argc;
		w = atof(*++argv);
		break;
	    case 'P':
		--argc;
		PRINT = atoi(*++argv);
		break;
	    default:
		printf("%s: ignored option: -%s\n", prog, *argv);
		printf("HELP: try %s -u \n\n", prog);
		break;
	    
		} 
}