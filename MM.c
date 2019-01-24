/***************************************************************************
 *
 * Parallel version of row-wise Matrix-Matrix multiplication
 * 
 *
 * BY
 * Akhil Tummala & Madhukar Enugurthi
 *
 *             
 ***************************************************************************/

/* 
 * Compile with:
 * mpicc -o mm mm.c 
 */
 
 /* 
 * Execute with:
 * mpirun -n 4 ./mm
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define SIZE 1024	/* assumption: SIZE a multiple of number of nodes */
#define FROM_MASTER 1	/* setting a message type */
#define FROM_WORKER 2	/* setting a message type */
#define DEBUG 0		/* 1 = debug on, 0 = debug off */

MPI_Status status;

static double a[SIZE][SIZE];
static double b[SIZE][SIZE];
static double c[SIZE][SIZE];

static void
init_matrix(void)
{
    int i, j;

    for (i = 0; i < SIZE; i++)
        for (j = 0; j < SIZE; j++) 
		{
	    a[i][j] = 1.0;
        if (i >= SIZE/2) 
			a[i][j] = 2.0;
        
		/* Matrix B Initialization with two dimensional array and it is the transpose. Since the matrix multiplication is more complex with columns,
						  so the matrix is transposed */
		b[j][i] = 1.0;
        if (j >= SIZE/2) 
			b[j][i] = 2.0; 
        }
}

static void
print_matrix(void)
{
    int i, j;

    for (i = 0; i < SIZE; i++) {
        for (j = 0; j < SIZE; j++)
	    printf(" %7.2f", c[i][j]);
	printf("\n");
    }
}

int
main(int argc, char **argv)
{
    int rank, nproc;
    int rows,  cols, rc; /* amount of work per node (rows per worker) */
    int mtype; 		/* message type: send/recv between master and workers */
    int dest, src, offset;
    double start_time, end_time;
    int i, j, k;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if(nproc == 1) /* node 1 code */
	{
		/* Initialization */
		printf("SIZE = %d, number of nodes = %d\n", SIZE, nproc);
		init_matrix();
		start_time = MPI_Wtime();
		
		for (i = 0; i < SIZE; i++)
			{
        		for (j = 0; j < SIZE; j++)
					{
						for (k = 0; k < SIZE; k++)
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}

			}
		
		end_time = MPI_Wtime();  /* ending the time*/
		if (DEBUG)
            /* Prints the resulting matrix c */
            print_matrix();
			
		
		printf("\n\nExecution time on %2d nodes: %f\n", nproc, end_time-start_time); /* printing the execution time */

	}
	else if(nproc == 2) /* node 2 code */
	{
		if(rank == 0)
		{
			/* Master task */
			
			/* Initialization */
			printf("SIZE = %d, number of nodes = %d\n", SIZE, nproc);
			init_matrix();
			start_time = MPI_Wtime();
			
			rows = SIZE/2;
			offset = rows;
			mtype = FROM_MASTER;
			
			/* Sending work for the worker 1*/
			MPI_Send(&rows, 1, MPI_INT, 1 , mtype, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, 1 , mtype, MPI_COMM_WORLD);
			MPI_Send(&a[offset][0], rows*SIZE, MPI_DOUBLE,1,mtype,MPI_COMM_WORLD);
			MPI_Send(&b, SIZE*SIZE, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD);
			
			
			/* Master work*/
			for (i = 0; i < offset; i++) 
			{
		        for (j = 0; j < SIZE; j++) 
				{
                    for (k = 0; k < SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}
                    
           		}
        	}
			
						
			/* Receving work data from worker*/
			mtype = FROM_WORKER;

			MPI_Recv(&offset, 1, MPI_INT, 1, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&rows, 1, MPI_INT, 1, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&c[offset][0], (SIZE*SIZE)/2, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
			
			end_time = MPI_Wtime();  /* ending the time*/
			if (DEBUG)
            /* Prints the resulting matrix c */
            print_matrix();
			
			
			printf("\n\nExecution time on %2d nodes: %f\n", nproc, end_time-start_time); /* printing the execution time */

		}
		else
		{
			/* Receving work from the master*/
			mtype = FROM_MASTER;
			MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&a[offset][0],rows*SIZE , MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
		    MPI_Recv(&b, SIZE*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			
			/* Worker's work*/
			for (i=offset; i<offset+rows; i++)
			{
				for (j=0; j<SIZE; j++)
				{
					
					for (k=0; k<SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}
				}
			}
			
			/* Worker Sending work result for Master */
			mtype = FROM_WORKER;
			MPI_Send(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
			MPI_Send(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD);
			MPI_Send(&c[offset][0], (SIZE*SIZE)/2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
			
		          
		}
	}
	else if (nproc == 4)
	{
		if(rank == 0)
		{
			/* Master task */
			
			/* Initialization */
			printf("SIZE = %d, number of nodes = %d\n", SIZE, nproc);
			init_matrix();
			start_time = MPI_Wtime();
			
			rows = SIZE/2; 
			offset =rows;
			cols =0;
			mtype = FROM_MASTER;
			
			/*Sending work for Worker 1 */
			MPI_Send(&rows, 1, MPI_INT, 1, mtype, MPI_COMM_WORLD);
			MPI_Send(&cols, 1, MPI_INT, 1, mtype, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, 1, mtype, MPI_COMM_WORLD);
			MPI_Send(&a[cols][0], rows*SIZE, MPI_DOUBLE,1,mtype,MPI_COMM_WORLD);
			MPI_Send(&b[offset][0], rows*SIZE, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD);
			
			/*sending work for Worker 2*/
			MPI_Send(&rows, 1, MPI_INT, 2, mtype, MPI_COMM_WORLD);
			MPI_Send(&cols, 1, MPI_INT, 2, mtype, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, 2, mtype, MPI_COMM_WORLD);
			MPI_Send(&a[offset][0], rows*SIZE, MPI_DOUBLE,2,mtype,MPI_COMM_WORLD);
			MPI_Send(&b[cols][0], rows*SIZE, MPI_DOUBLE, 2, mtype, MPI_COMM_WORLD);
			
			/*sending work for Worker 3*/
			MPI_Send(&rows, 1, MPI_INT, 3, mtype, MPI_COMM_WORLD);
			MPI_Send(&cols, 1, MPI_INT, 3, mtype, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, 3, mtype, MPI_COMM_WORLD);
			MPI_Send(&a[offset][0], rows*SIZE, MPI_DOUBLE,3,mtype,MPI_COMM_WORLD);
			MPI_Send(&b[offset][0],rows*SIZE, MPI_DOUBLE, 3, mtype, MPI_COMM_WORLD);
			
			/* Master work */
			for (i = 0; i < offset; i++) 
			{
		        for (j = 0; j < offset; j++) 
				{
                    for (k = 0; k < SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}
                    
           		}
        	}
			
			/*Receving from worker 1*/
			for (i=0; i<offset; i++)
			{
				mtype = FROM_WORKER;
				MPI_Recv(&c[i][offset],SIZE/2, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
			}
			
			/*Receving from worker 2*/
			for (i=offset; i<SIZE; i++)
			{
				mtype = FROM_WORKER;
				MPI_Recv(&c[i][0],SIZE/2, MPI_DOUBLE, 2, mtype, MPI_COMM_WORLD, &status);
			}
			
			/*Receving from worker 3*/
			for (i=offset; i<SIZE; i++)
			{
				mtype = FROM_WORKER;
				MPI_Recv(&c[i][offset],SIZE/2, MPI_DOUBLE, 3, mtype, MPI_COMM_WORLD, &status);
			}	
			
			end_time = MPI_Wtime();  /* ending the time*/
			if (DEBUG)
            /* Prints the resulting matrix c */
            print_matrix();
			
			
			printf("\n\nExecution time on %2d nodes: %f\n", nproc, end_time-start_time); /* printing the execution time */

			
		}
		else if(rank == 1)
		{
			/*Worker Receving work from master */
			mtype = FROM_MASTER;
			MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&a[cols][0],rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b[offset][0],rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			
			/* Worker's Work */
			for (i=0; i<offset; i++)
			{
				for (j=offset; j<SIZE; j++)
				{					
					for (k=0; k<SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}				
				}
				/* Worker sending work result to master */
				mtype = FROM_WORKER;
				MPI_Send(&c[i][offset],SIZE/2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);						
			}			
		}
		else if(rank == 2)
		{
			/*Worker Receving work from master */
			mtype = FROM_MASTER;
			MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&a[offset][0],rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b[cols][0], rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
		    
			/* Worker's Work */
			for (i=offset; i<SIZE; i++)
			{
				for (j=0; j<offset; j++)
				{					
					for (k=0; k<SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}
				
				}
				/* Worker sending work result to master */
				mtype = FROM_WORKER;			
				MPI_Send(&c[i][0], SIZE/2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
			}			
		}
		else if(rank == 3)
		{
			/*Worker Receving work from master */
			mtype = FROM_MASTER;
			MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&a[offset][0],rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b[offset][0], rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
		    
			/* Worker's Work */
			for (i=offset; i<SIZE; i++)
			{
				for (j=offset; j<SIZE; j++)
				{
					for (k=0; k<SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}					
				}
				/* Worker sending work result to master */
				mtype = FROM_WORKER;
				MPI_Send(&c[i][offset],SIZE/2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
			}			
		}
	}
	
	else if (nproc == 8)
	{
		
		if(rank == 0)
		{
			/* Master task */
			
			/* Initialization */
			printf("SIZE = %d, number of nodes = %d\n", SIZE, nproc);
			init_matrix();
			start_time = MPI_Wtime();
			
			rows = SIZE/2; 
			cols = SIZE/4;
			rc = rows + cols;
			offset = 0 ;
			mtype = FROM_MASTER;
			
			/*sending work for Worker 1*/
			MPI_Send(&rows, 1, MPI_INT, 1, mtype, MPI_COMM_WORLD);
			MPI_Send(&cols, 1, MPI_INT, 1, mtype, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, 1, mtype, MPI_COMM_WORLD);
			MPI_Send(&a[offset][0], cols*SIZE, MPI_DOUBLE,1,mtype,MPI_COMM_WORLD);
			MPI_Send(&b[rows][0], rows*SIZE, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD);
			
			/*sending work for Worker 2*/
			MPI_Send(&rows, 1, MPI_INT, 2, mtype, MPI_COMM_WORLD);
			MPI_Send(&cols, 1, MPI_INT, 2, mtype, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, 2, mtype, MPI_COMM_WORLD);
			MPI_Send(&a[cols][0], cols*SIZE, MPI_DOUBLE,2,mtype,MPI_COMM_WORLD);
			MPI_Send(&b[offset][0], rows*SIZE, MPI_DOUBLE, 2, mtype, MPI_COMM_WORLD);
			
			/*sending work for Worker 3*/
			MPI_Send(&rows, 1, MPI_INT, 3, mtype, MPI_COMM_WORLD);
			MPI_Send(&cols, 1, MPI_INT, 3, mtype, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, 3, mtype, MPI_COMM_WORLD);
			MPI_Send(&a[cols][0], cols*SIZE, MPI_DOUBLE,3,mtype,MPI_COMM_WORLD);
			MPI_Send(&b[rows][0], rows*SIZE, MPI_DOUBLE, 3, mtype, MPI_COMM_WORLD);
			
			/*sending work for Worker 4*/
			MPI_Send(&rows, 1, MPI_INT, 4, mtype, MPI_COMM_WORLD);
			MPI_Send(&cols, 1, MPI_INT, 4, mtype, MPI_COMM_WORLD);
			MPI_Send(&rc, 1, MPI_INT, 4, mtype, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, 4, mtype, MPI_COMM_WORLD);
			MPI_Send(&a[rows][0], cols*SIZE, MPI_DOUBLE,4,mtype,MPI_COMM_WORLD);
			MPI_Send(&b[offset][0], rows*SIZE, MPI_DOUBLE, 4, mtype, MPI_COMM_WORLD);
			
			/*sending work for Worker 5*/
			MPI_Send(&rows, 1, MPI_INT, 5, mtype, MPI_COMM_WORLD);
			MPI_Send(&cols, 1, MPI_INT, 5, mtype, MPI_COMM_WORLD);
			MPI_Send(&rc, 1, MPI_INT, 5, mtype, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, 5, mtype, MPI_COMM_WORLD);
			MPI_Send(&a[rows][0], cols*SIZE, MPI_DOUBLE,5,mtype,MPI_COMM_WORLD);
			MPI_Send(&b[rows][0], rows*SIZE, MPI_DOUBLE, 5, mtype, MPI_COMM_WORLD);
			
			/*sending work for Worker 6*/
			MPI_Send(&rows, 1, MPI_INT, 6, mtype, MPI_COMM_WORLD);
			MPI_Send(&cols, 1, MPI_INT, 6, mtype, MPI_COMM_WORLD);
			MPI_Send(&rc, 1, MPI_INT, 6, mtype, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, 6, mtype, MPI_COMM_WORLD);
			MPI_Send(&a[rc][0], cols*SIZE, MPI_DOUBLE,6,mtype,MPI_COMM_WORLD);
			MPI_Send(&b[offset][0], rows*SIZE, MPI_DOUBLE, 6, mtype, MPI_COMM_WORLD);
			
			/*sending work for Worker 7*/
			MPI_Send(&rows, 1, MPI_INT, 7, mtype, MPI_COMM_WORLD);
			MPI_Send(&cols, 1, MPI_INT, 7, mtype, MPI_COMM_WORLD);
			MPI_Send(&rc, 1, MPI_INT, 7, mtype, MPI_COMM_WORLD);
			MPI_Send(&offset, 1, MPI_INT, 7, mtype, MPI_COMM_WORLD);
			MPI_Send(&a[rc][0], cols*SIZE, MPI_DOUBLE,7,mtype,MPI_COMM_WORLD);
			MPI_Send(&b[rows][0], rows*SIZE, MPI_DOUBLE, 7, mtype, MPI_COMM_WORLD);
						
			/* Master's Work */
			for (i=0; i<cols; i++)
			{
				for (j=0; j<rows; j++)
				{					
					for (k=0; k<SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}
				
				}
			}
			
			/*Receving from worker 1*/
			for (i=0; i<cols; i++)
			{
				mtype = FROM_WORKER;
				MPI_Recv(&c[i][rows],SIZE/2, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
			}
			
			/*Receving from worker 2*/
			for (i=cols; i<rows; i++)
			{
				mtype = FROM_WORKER;
				MPI_Recv(&c[i][0],SIZE/2, MPI_DOUBLE, 2, mtype, MPI_COMM_WORLD, &status);
			}
			
			/*Receving from worker 3*/
			for (i=cols; i< rows; i++)
			{
				mtype = FROM_WORKER;
				MPI_Recv(&c[i][rows],SIZE/2, MPI_DOUBLE, 3, mtype, MPI_COMM_WORLD, &status);
			}
			
			/*Receving from worker 4*/
			for (i=rows; i< rc; i++)
			{
				mtype = FROM_WORKER;
				MPI_Recv(&c[i][0],SIZE/2, MPI_DOUBLE, 4, mtype, MPI_COMM_WORLD, &status);
			}
			
			/*Receving from worker 5*/
			for (i=rows; i< rc; i++)
			{
				mtype = FROM_WORKER;
				MPI_Recv(&c[i][rows],SIZE/2, MPI_DOUBLE, 5, mtype, MPI_COMM_WORLD, &status);
			}
			
			/*Receving from worker 6*/
			for (i=rc; i< SIZE; i++)
			{
				mtype = FROM_WORKER;
				MPI_Recv(&c[i][0],SIZE/2, MPI_DOUBLE, 6, mtype, MPI_COMM_WORLD, &status);
			}
			
			/*Receving from worker 7*/
			for (i=rc; i<SIZE; i++)
			{
				mtype = FROM_WORKER;
				MPI_Recv(&c[i][rows],SIZE/2, MPI_DOUBLE, 7, mtype, MPI_COMM_WORLD, &status);
			}
			
			end_time = MPI_Wtime();  /* ending the time*/
			
			if (DEBUG)
            /* Prints the resulting matrix c */
            print_matrix();
			
			printf("\n\nExecution time on %2d nodes: %f\n", nproc, end_time-start_time); /* printing the execution time */
			
							
		}
		else if(rank == 1)
		{
			/*Worker Receving work from master */
			mtype = FROM_MASTER;
			MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&a[offset][0],cols*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b[rows][0], rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			
			/* Worker's Work */
			for (i=0; i<cols; i++)
			{
				for (j=rows; j<SIZE; j++)
				{
					for (k=0; k<SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}					
				}
				
				/* Worker sending work result to master */
				mtype = FROM_WORKER;
				MPI_Send(&c[i][rows],SIZE/2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
			}	
			
		}
		else if(rank == 2)
		{
			/*Worker Receving work from master */
			mtype = FROM_MASTER;
			MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&a[cols][0],cols*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b[offset][0], rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			
			/* Worker's Work */
			for (i= cols; i<rows; i++)
			{
				for (j= 0 ; j< rows; j++)
				{
					for (k=0; k<SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}					
				}
				/* Worker sending work result to master */
				mtype = FROM_WORKER;
				MPI_Send(&c[i][0],SIZE/2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
			}	
		}
		else if(rank == 3)
		{
			/*Worker Receving work from master */
			mtype = FROM_MASTER;
			MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&a[cols][0],cols*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b[rows][0], rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			
			/* Worker's Work */
			for (i= cols; i< rows; i++)
			{
				for (j=rows; j<SIZE; j++)
				{
					for (k=0; k<SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}					
				}
				/* Worker sending work result to master */
				mtype = FROM_WORKER;
				MPI_Send(&c[i][rows],SIZE/2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
			}	
		}
		else if(rank == 4)
		{
			/*Worker Receving work from master */
			mtype = FROM_MASTER;
			MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&rc, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&a[rows][0],cols*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b[offset][0], rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			
			/* Worker's Work */
			for (i= rows; i< rc ; i++)
			{
				for (j=0; j< rows; j++)
				{
					for (k=0; k<SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}					
				}
				/* Worker sending work result to master */
				mtype = FROM_WORKER;
				MPI_Send(&c[i][0],SIZE/2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
			}	
		}
		else if(rank == 5)
		{
			/*Worker Receving work from master */
			mtype = FROM_MASTER;
			MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&rc, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&a[rows][0],cols*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b[rows][0], rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			
			/* Worker's Work */
			for (i=rows; i< rc; i++)
			{
				for (j= rows; j<SIZE; j++)
				{
					for (k=0; k<SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}					
				}
				/* Worker sending work result to master */
				mtype = FROM_WORKER;
				MPI_Send(&c[i][rows],SIZE/2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
			}	
			
		}
		else if(rank == 6)
		{
			/*Worker Receving work from master */
			mtype = FROM_MASTER;
			MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&rc, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&a[rc][0],cols*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b[offset][0], rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			
			/* Worker's Work */
			for (i= rc; i<SIZE; i++)
			{
				for (j= 0; j<rows; j++)
				{
					for (k=0; k<SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}					
				}
				/* Worker sending work result to master */
				mtype = FROM_WORKER;
				MPI_Send(&c[i][0],SIZE/2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
			}	
		}
		else if(rank == 7)
		{
			/*Worker Receving work from master */
			mtype = FROM_MASTER;
			MPI_Recv(&rows, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&cols, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&rc, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&offset, 1, MPI_INT, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&a[rc][0],cols*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			MPI_Recv(&b[rows][0], rows*SIZE, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD, &status);
			
			/* Worker's Work */
			for (i= rc; i<SIZE; i++)
			{
				for (j=rows; j<SIZE; j++)
				{
					for (k=0; k<SIZE; k++)
					{
						c[i][j] = c[i][j] + a[i][k] * b[j][k];
					}					
				}
				/* Worker sending work result to master */
				mtype = FROM_WORKER;
				MPI_Send(&c[i][rows],SIZE/2, MPI_DOUBLE, 0, mtype, MPI_COMM_WORLD);
			}	
		}
	}
					
    MPI_Finalize();
    return 0; 
}