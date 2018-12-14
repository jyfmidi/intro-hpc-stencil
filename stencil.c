
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "mpi.h"

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(const int nx, const int ny, float *image, float *tmp_image);

void init_image(const int nx, const int ny, float *image, float *tmp_image);

void output_image(const char *file_name, const int nx, const int ny, float *image);

double wtime(void);

int calc_ncols_from_rank(int rank, int size, int nx);

int main(int argc, char *argv[]) {
    int ii, jj;             /* row and column indices for the grid */
    int kk;                /* index for looping over ranks */
    int rank;              /* the rank of this process */
    int up;              /* the rank of the process to the left */
    int down;             /* the rank of the process to the right */
    int size;              /* number of processes in the communicator */
    int tag = 0;           /* scope for adding extra information to a message */
    MPI_Status status;     /* struct used by MPI_Recv */
    int local_nrows;       /* number of rows apportioned to this rank */
    int local_ncols;       /* number of columns apportioned to this rank */
    //int remote_nrows;      /* number of columns apportioned to a remote rank */
    float *w;             /* local temperature grid at time t     */
    float *u;             /* local temperature grid at time t     */
    float *sendbuf;       /* buffer to hold values to send */
    float *recvbuf;       /* buffer to hold received values */
    float *printbuf;      /* buffer to hold values for printing */

    /*
    ** MPI_Init returns once it has started up processes
    ** Get size of cohort and rank for this process
    */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initiliase problem dimensions from command line arguments
    int nx = atoi(argv[1]);
    int ny = atoi(argv[2]);
    int niters = atoi(argv[3]);
    // Allocate the image
    float *image = malloc(sizeof(float) * nx * ny);
    float *tmp_image = malloc(sizeof(float) * nx * ny);

    // Set the input image
    init_image(nx, ny, image, tmp_image);


    /*
    ** determine process ranks to the left and right of rank
    ** respecting periodic boundary conditions
    */
    up = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
    down = (rank + 1) % size;

    /*
    ** determine local grid size
    ** each rank gets all the rows, but a subset of the number of columns
    */
    local_nrows = calc_ncols_from_rank(rank, size, nx);
    local_ncols = nx;
    if (local_nrows < 1) {
        fprintf(stderr, "Error: too many processes:- local_ncols < 1\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /*
    ** allocate space for:
    ** - the local grid with 2 extra columns added for the halos
    ** - message passing buffers
    ** - a buffer used to print local grid values
    */
    w = (float *) malloc(sizeof(float *) * (local_nrows + 2) * local_ncols);
    u = (float *) malloc(sizeof(float *) * (local_nrows + 2) * local_ncols);
    sendbuf = (float *) malloc(sizeof(float) * local_ncols);
    recvbuf = (float *) malloc(sizeof(float) * local_ncols);
    /* The last rank has the most columns apportioned.
       printbuf must be big enough to hold this number */
    printbuf = (float *) malloc(sizeof(float) * local_nrows * local_ncols);

    /*
    ** initialize the local grid (w):
    ** - core cells are set to the value of the rank
    ** - halo cells are inititalised to a -ve value
    ** note the looping bounds for index jj is modified
    ** to accomodate the extra halo columns
    */
    for (jj = 0; jj < local_ncols; jj++)
        w[jj] = 0;
    for (ii = 1; ii < local_nrows + 1; ii++) {
        for (jj = 0; jj < local_ncols; jj++) {
            w[ii * local_ncols + jj] = image[(rank * (nx / size) + ii - 1) * local_ncols + jj];
        }
    }
    for (jj = 0; jj < local_ncols; jj++)
        w[(local_nrows + 1) * local_ncols + jj] = 0;

    // ## PHASE 1: w->u
    /* send to the up, receive from down */
    for (jj = 0; jj < local_ncols; jj++)
        sendbuf[jj] = w[local_ncols + jj];
    MPI_Sendrecv(sendbuf, local_ncols, MPI_FLOAT, up, tag,
                 recvbuf, local_ncols, MPI_FLOAT, down, tag,
                 MPI_COMM_WORLD, &status);
    if (rank < size - 1) {
        for (jj = 0; jj < local_ncols; jj++)
            w[(local_nrows + 1) * local_ncols + jj] = recvbuf[jj];
    } else {
        for (jj = 0; jj < local_ncols; jj++)
            w[(local_nrows + 1) * local_ncols + jj] = 0;
    }
    /* send to the down, receive from up */
    for (jj = 0; jj < local_ncols; jj++)
        sendbuf[jj] = w[local_nrows * local_ncols + jj];
    MPI_Sendrecv(sendbuf, local_ncols, MPI_FLOAT, down, tag,
                 recvbuf, local_ncols, MPI_FLOAT, up, tag,
                 MPI_COMM_WORLD, &status);
    if (rank > 0) {
        for (jj = 0; jj < local_ncols; jj++)
            w[jj] = recvbuf[jj];
    } else {
        for (jj = 0; jj < local_ncols; jj++)
            w[jj] = 0;
    }

    double tic = wtime();
    for (int iter = 0; iter < niters; iter++) {
        stencil(local_nrows + 2, local_ncols, w, u);

        for (jj = 0; jj < local_ncols; jj++)
            sendbuf[jj] = u[local_ncols + jj];
        MPI_Sendrecv(sendbuf, local_ncols, MPI_FLOAT, up, tag,
                     recvbuf, local_ncols, MPI_FLOAT, down, tag,
                     MPI_COMM_WORLD, &status);
        if (rank < size - 1) {
            for (jj = 0; jj < local_ncols; jj++)
                u[(local_nrows + 1) * local_ncols + jj] = recvbuf[jj];
        } else {
            for (jj = 0; jj < local_ncols; jj++)
                u[(local_nrows + 1) * local_ncols + jj] = 0;
        }
        /* send to the down, receive from up */
        for (jj = 0; jj < local_ncols; jj++)
            sendbuf[jj] = u[local_nrows * local_ncols + jj];
        MPI_Sendrecv(sendbuf, local_ncols, MPI_FLOAT, down, tag,
                     recvbuf, local_ncols, MPI_FLOAT, up, tag,
                     MPI_COMM_WORLD, &status);
        if (rank > 0) {
            for (jj = 0; jj < local_ncols; jj++)
                u[jj] = recvbuf[jj];
        } else {
            for (jj = 0; jj < local_ncols; jj++)
                u[jj] = 0;
        }

        stencil(local_nrows + 2, local_ncols, u, w);

        for (jj = 0; jj < local_ncols; jj++)
            sendbuf[jj] = w[local_ncols + jj];
        MPI_Sendrecv(sendbuf, local_ncols, MPI_FLOAT, up, tag,
                     recvbuf, local_ncols, MPI_FLOAT, down, tag,
                     MPI_COMM_WORLD, &status);
        if (rank < size - 1) {
            for (jj = 0; jj < local_ncols; jj++)
                w[(local_nrows + 1) * local_ncols + jj] = recvbuf[jj];
        } else {
            for (jj = 0; jj < local_ncols; jj++)
                w[(local_nrows + 1) * local_ncols + jj] = 0;
        }
        /* send to the down, receive from up */
        for (jj = 0; jj < local_ncols; jj++)
            sendbuf[jj] = w[local_nrows * local_ncols + jj];
        MPI_Sendrecv(sendbuf, local_ncols, MPI_FLOAT, down, tag,
                     recvbuf, local_ncols, MPI_FLOAT, up, tag,
                     MPI_COMM_WORLD, &status);
        if (rank > 0) {
            for (jj = 0; jj < local_ncols; jj++)
                w[jj] = recvbuf[jj];
        } else {
            for (jj = 0; jj < local_ncols; jj++)
                w[jj] = 0;
        }
    }

    double toc = wtime();

    if (rank == MASTER) {
        printf("master here");
        printf("------------------------------------\n");
        printf(" runtime: %lf s\n", toc - tic);
        printf("------------------------------------\n");
    }

    // // ignore the halo

    if (rank == MASTER){
        for (ii = 1; ii < local_nrows + 1; ii++) {
            for (jj = 0; jj < local_ncols; jj++) {
                tmp_image[(ii-1)*local_ncols+jj] = w[ii * local_ncols + jj];
            }
        }
        for(kk=1;kk<size;kk++){
            MPI_Recv(printbuf,local_nrows*local_ncols,MPI_FLOAT,kk,tag, MPI_COMM_WORLD, &status);
            for(int i=0;i<local_nrows;i++){
                for(int j=0;j<local_ncols;j++){
                    tmp_image[(kk*(nx/size)+i)*local_ncols+j] = printbuf[i*local_ncols+j];
                }
            }
        }
    } else {
        MPI_Send(&w[local_ncols],local_nrows*local_ncols,MPI_FLOAT,MASTER,tag, MPI_COMM_WORLD);
    }
    

    if (rank == MASTER) {
        output_image(OUTPUT_FILE, nx, ny, tmp_image);
    }
    // if(rank == MASTER) {
    //     for(int i=0;i<nx;i++){
    //         for(int j=0;j<ny;j++)
    //             printf("%2.1f \t", tmp_image[i*ny+j]);
    //         printf("\n");
    //     }
    // }
    
    /* don't forget to tidy up when we're done */
    MPI_Finalize();
    free(w);
    free(u);
    free(sendbuf);
    free(recvbuf);
    free(printbuf);
    /* and exit the program */
    return EXIT_SUCCESS;
}

// void stencil(const int nx, const int ny, float * restrict image, float * restrict tmp_image) {
// #pragma omp parallel for
//   for (int i = 0; i < nx; ++i) {
//     #pragma omp simd
//     for (int j = 0; j < ny; ++j) {
//       float temp = 0;
//       temp = image[j+i*ny] * 0.6f;
//       if (j > 0)    temp += image[j-1+i*ny] * 0.1f;
//       if (j < ny-1) temp += image[j+1+i*ny] * 0.1f;
//       if (i > 0)    temp += image[j  +(i-1)*ny] * 0.1f;
//       if (i < nx-1) temp += image[j  +(i+1)*ny] * 0.1f;
//       tmp_image[j+i*ny] = temp;
//     }
//   }
// }

void stencil(const int nx, const int ny, float *restrict image, float *restrict tmp_image) {
    //i=0;j=0, no left, no up
    tmp_image[0] = image[0] * 0.6f + image[1] * 0.1f + image[ny] * 0.1f;
    //top edge, i=0, no up
    for (int j = 1; j < ny - 1; ++j)
        tmp_image[j] = image[j] * 0.6f + image[j - 1] * 0.1f + image[j + 1] * 0.1f + image[j + ny] * 0.1f;
    //i=0;j=ny-1, no right, no up
    tmp_image[ny - 1] = image[ny - 1] * 0.6f + image[ny - 2] * 0.1f + image[ny + ny - 1] * 0.1f;

    for (int i = 1; i < nx - 1; ++i) {
        //left pixel, j=0, no left
        tmp_image[i * ny] = image[i * ny] * 0.6f + image[1 + i * ny] * 0.1f + image[(i - 1) * ny] * 0.1f +
                            image[(i + 1) * ny] * 0.1f;
        for (int j = 1; j < ny - 1; ++j) {
            tmp_image[j + i * ny] =
                    image[j + i * ny] * 0.6f + image[j - 1 + i * ny] * 0.1f + image[j + 1 + i * ny] * 0.1f +
                    image[j + (i - 1) * ny] * 0.1f + image[j + (i + 1) * ny] * 0.1f;
        }
        //right pixel, j=ny-1, no right
        tmp_image[ny - 1 + i * ny] =
                image[ny - 1 + i * ny] * 0.6f + image[ny - 2 + i * ny] * 0.1f + image[ny - 1 + (i - 1) * ny] * 0.1f +
                image[ny - 1 + (i + 1) * ny] * 0.1f;

    }
    //i=nx-1;j=0, no left, no down
    tmp_image[(nx - 1) * ny] =
            image[(nx - 1) * ny] * 0.6f + image[(nx - 1) * ny + 1] * 0.1f + image[(nx - 2) * ny] * 0.1f;
    //bottom edge, i=nx-1, no bottom
    for (int j = 1; j < ny - 1; ++j)
        tmp_image[j + (nx - 1) * ny] = image[j + (nx - 1) * ny] * 0.6f + image[j - 1 + (nx - 1) * ny] * 0.1f +
                                       image[j + 1 + (nx - 1) * ny] * 0.1f + image[j + (nx - 2) * ny] * 0.1f;
    //i=nx-1.j=ny-1, no right, no down
    tmp_image[(nx - 1) * ny + (ny - 1)] =
            image[(nx - 1) * ny + (ny - 1)] * 0.6f + image[(nx - 1) * ny + (ny - 2)] * 0.1f +
            image[(nx - 2) * ny + (ny - 1)] * 0.1f;
}

// Create the input image
void init_image(const int nx, const int ny, float *image, float *tmp_image) {
    // Zero everything
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            image[j + i * ny] = 0.0f;
            tmp_image[j + i * ny] = 0.0f;
        }
    }

    // Checkerboard
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            for (int jj = j * ny / 8; jj < (j + 1) * ny / 8; ++jj) {
                for (int ii = i * nx / 8; ii < (i + 1) * nx / 8; ++ii) {
                    if ((i + j) % 2)
                        image[jj + ii * ny] = 100.0f;
                }
            }
        }
    }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char *file_name, const int nx, const int ny, float *image) {
    // Open output file
    FILE *fp = fopen(file_name, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
        exit(EXIT_FAILURE);
    }

    // Ouptut image header
    fprintf(fp, "P5 %d %d 255\n", nx, ny);

    // Calculate maximum value of image
    // This is used to rescale the values
    // to a range of 0-255 for output
    float maximum = 0.0f;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            if (image[j + i * ny] > maximum)
                maximum = image[j + i * ny];
        }
    }
    // Output image, converting to numbers 0-255
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            fputc((char) (255.0f * image[j + i * ny] / maximum), fp);
        }
    }

    // Close the file
    fclose(fp);

}

// Get the current time in seconds since the Epoch
double wtime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int calc_ncols_from_rank(int rank, int size, int nx) {
    int nrows;

    nrows = nx / size;       /* integer division */
    if ((nx % size) != 0) {  /* if there is a remainder */
        if (rank == size - 1)
            nrows += nx % size;  /* add remainder to last rank */
    }

    return nrows;
}

