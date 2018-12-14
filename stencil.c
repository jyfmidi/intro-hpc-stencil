
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

int calc_ncols_from_rank(int rank, int size, int ny);

int main(int argc, char *argv[]) {
    // Initiliase problem dimensions from command line arguments
    int ii, jj;             /* row and column indices for the grid */
    int kk;                /* index for looping over ranks */
    int rank;              /* the rank of this process */
    int left;              /* the rank of the process to the left */
    int right;             /* the rank of the process to the right */
    int size;              /* number of processes in the communicator */
    int tag = 0;           /* scope for adding extra information to a message */
    int tag1 = 1;
    MPI_Status status;     /* struct used by MPI_Recv */
    int local_nrows;       /* number of rows apportioned to this rank */
    int local_ncols;       /* number of columns apportioned to this rank */
    int remote_ncols;      /* number of columns apportioned to a remote rank */
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
    left = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
    right = (rank + 1) % size;

    /*
    ** determine local grid size
    ** each rank gets all the rows, but a subset of the number of columns
    */
    local_nrows = nx;
    local_ncols = calc_ncols_from_rank(rank, size, ny);
    if (local_ncols < 1) {
        fprintf(stderr, "Error: too many processes:- local_ncols < 1\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    /*
    ** allocate space for:
    ** - the local grid with 2 extra columns added for the halos
    ** - message passing buffers
    ** - a buffer used to print local grid values
    */
    w = (float *) malloc(sizeof(float *) * local_nrows * (local_ncols + 2));
    u = (float *) malloc(sizeof(float *) * local_nrows * (local_ncols + 2));
    sendbuf = (float *) malloc(sizeof(float) * local_nrows);
    recvbuf = (float *) malloc(sizeof(float) * local_nrows);
    /* The last rank has the most columns apportioned.
       printbuf must be big enough to hold this number */
    remote_ncols = calc_ncols_from_rank(size - 1, size, ny);
    printbuf = (float *) malloc(sizeof(float) * (remote_ncols + 2));

    /*
    ** initialize the local grid (w):
    ** - core cells are set to the value of the rank
    ** - halo cells are inititalised to a -ve value
    ** note the looping bounds for index jj is modified
    ** to accomodate the extra halo columns
    */
    for (ii = 0; ii < local_nrows; ii++) {
        for (jj = 0; jj < local_ncols + 2; jj++) {
            if (jj > 0 && jj < (local_ncols + 1))
                w[ii * (local_ncols + 2) + jj] = image[ii * ny + jj - 1 +
                                                       rank * (ny / size)];                 /* core cells */
            else if (jj == 0 || jj == (local_ncols + 1))
                w[ii * (local_ncols + 2) + jj] = 0;                         /* halo cells */
        }
    }

    /*
    ** halo exchange for the local grids w:
    ** - first send to the left and receive from the right,
    ** - then send to the right and receive from the left.
    ** for each direction:
    ** - first, pack the send buffer using values from the grid
    ** - exchange using MPI_Sendrecv()
    ** - unpack values from the recieve buffer into the grid
    */

    // ## PHASE 1: w->u
    /* send to the left, receive from right */
    for (ii = 0; ii < local_nrows; ii++)
        sendbuf[ii] = w[ii * (local_ncols + 2) + 1];
    MPI_Sendrecv(sendbuf, local_nrows, MPI_FLOAT, left, tag,
                 recvbuf, local_nrows, MPI_FLOAT, right, tag,
                 MPI_COMM_WORLD, &status);
    if (rank < size - 1) {
        for (ii = 0; ii < local_nrows; ii++)
            w[ii * (local_ncols + 2) + local_ncols + 1] = recvbuf[ii];
    } else {
        for (ii = 0; ii < local_nrows; ii++)
            w[ii * (local_ncols + 2) + local_ncols + 1] = 0;
    }
    /* send to the right, receive from left */
    for (ii = 0; ii < local_nrows; ii++)
        sendbuf[ii] = w[ii * (local_ncols + 2) + local_ncols];
    MPI_Sendrecv(sendbuf, local_nrows, MPI_FLOAT, right, tag,
                 recvbuf, local_nrows, MPI_FLOAT, left, tag,
                 MPI_COMM_WORLD, &status);
    if (rank > 0) {
        for (ii = 0; ii < local_nrows; ii++)
            w[ii * (local_ncols + 2)] = recvbuf[ii];
    } else {
        for (ii = 0; ii < local_nrows; ii++)
            w[ii * (local_ncols + 2)] = 0;
    }


    double tic = wtime();

    for (int iter = 0; iter < niters; iter++) {
        stencil(local_nrows, local_ncols + 2, w, u);

        // ## PHASE 2 u->w
        // swapping after first computing
        for (ii = 0; ii < local_nrows; ii++)
            sendbuf[ii] = u[ii * (local_ncols + 2) + 1];
        MPI_Sendrecv(sendbuf, local_nrows, MPI_FLOAT, left, tag,
                     recvbuf, local_nrows, MPI_FLOAT, right, tag,
                     MPI_COMM_WORLD, &status);
        if (rank < size - 1) {
            for (ii = 0; ii < local_nrows; ii++)
                u[ii * (local_ncols + 2) + local_ncols + 1] = recvbuf[ii];
        } else {
            for (ii = 0; ii < local_nrows; ii++)
                u[ii * (local_ncols + 2) + local_ncols + 1] = 0;
        }
        /* send to the right, receive from left */
        for (ii = 0; ii < local_nrows; ii++)
            sendbuf[ii] = u[ii * (local_ncols + 2) + local_ncols];
        MPI_Sendrecv(sendbuf, local_nrows, MPI_FLOAT, right, tag,
                     recvbuf, local_nrows, MPI_FLOAT, left, tag,
                     MPI_COMM_WORLD, &status);
        if (rank > 0) {
            for (ii = 0; ii < local_nrows; ii++)
                u[ii * (local_ncols + 2)] = recvbuf[ii];
        } else {
            for (ii = 0; ii < local_nrows; ii++)
                u[ii * (local_ncols + 2)] = 0;
        }

        stencil(local_nrows, local_ncols + 2, u, w);

        for (ii = 0; ii < local_nrows; ii++)
            sendbuf[ii] = w[ii * (local_ncols + 2) + 1];
        MPI_Sendrecv(sendbuf, local_nrows, MPI_FLOAT, left, tag,
                     recvbuf, local_nrows, MPI_FLOAT, right, tag,
                     MPI_COMM_WORLD, &status);
        if (rank < size - 1) {
            for (ii = 0; ii < local_nrows; ii++)
                w[ii * (local_ncols + 2) + local_ncols + 1] = recvbuf[ii];
        } else {
            for (ii = 0; ii < local_nrows; ii++)
                w[ii * (local_ncols + 2) + local_ncols + 1] = 0;
        }
        /* send to the right, receive from left */
        for (ii = 0; ii < local_nrows; ii++)
            sendbuf[ii] = w[ii * (local_ncols + 2) + local_ncols];
        MPI_Sendrecv(sendbuf, local_nrows, MPI_FLOAT, right, tag,
                     recvbuf, local_nrows, MPI_FLOAT, left, tag,
                     MPI_COMM_WORLD, &status);
        if (rank > 0) {
            for (ii = 0; ii < local_nrows; ii++)
                w[ii * (local_ncols + 2)] = recvbuf[ii];
        } else {
            for (ii = 0; ii < local_nrows; ii++)
                w[ii * (local_ncols + 2)] = 0;
        }

    }

    double toc = wtime();

    if (rank == MASTER) {
        printf("master here");
        printf("--------------\n");
        printf(" runtime: %lf s\n", toc - tic);
        printf("--------------\n");
    }

    // ignore the halo
    int idx = 0;
    for (ii = 0; ii < local_nrows; ii++) {
        if (rank == MASTER) {
            for (jj = 1; jj < local_ncols + 1; jj++) {
                tmp_image[idx++] = w[ii * (local_ncols + 2) + jj];
            }
            for (kk = 1; kk < size; kk++) { /* loop over other ranks */
                remote_ncols = calc_ncols_from_rank(kk, size,ny);
                MPI_Recv(printbuf, remote_ncols + 2, MPI_FLOAT, kk, tag1, MPI_COMM_WORLD, &status);
                for (jj = 1; jj < remote_ncols + 1; jj++) {
                    tmp_image[idx++] = printbuf[jj];
                }
            }
        } else {
            MPI_Send(&w[ii * (local_ncols + 2)], local_ncols + 2, MPI_FLOAT, MASTER, tag1, MPI_COMM_WORLD);
        }
    }
    if (rank == MASTER) {
        output_image(OUTPUT_FILE, nx, ny, tmp_image);
    }

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

int calc_ncols_from_rank(int rank, int size, int ny) {
    int ncols;

    ncols = ny / size;       /* integer division */
    if ((ny % size) != 0) {  /* if there is a remainder */
        if (rank == size - 1)
            ncols += ny % size;  /* add remainder to last rank */
    }

    return ncols;
}

