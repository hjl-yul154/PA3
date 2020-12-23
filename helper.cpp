/* 
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#include <mpi.h>
#include "apf.h"
#include "cblock.h"



using namespace std;

double *alloc1D(int m,int n);
void printMat(const char mesg[], double *E, int m, int n);
void distribute(double *E,double *E_prev,double *R,int m,int n);
extern control_block cb;
//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init (double *E,double *E_prev,double *R,int m,int n){
    int i;

    for (i=0; i < (m+2)*(n+2); i++)
        E_prev[i] = R[i] = 0;

    for (i = (n+2); i < (m+1)*(n+2); i++) {
    int colIndex = i % (n+2);       // gives the base index (first row's) of the current index

        // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
    if(colIndex == 0 || colIndex == (n+1) || colIndex < ((n+1)/2+1))
        continue;

        E_prev[i] = 1.0;
    }

    for (i = 0; i < (m+2)*(n+2); i++) {
    int rowIndex = i / (n+2);       // gives the current row number in 2D array representation
    int colIndex = i % (n+2);       // gives the base index (first row's) of the current index

        // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
    if(colIndex == 0 || colIndex == (n+1) || rowIndex < ((m+1)/2+1))
        continue;

        R[i] = 1.0;
    }
#ifdef _MPI_
    distribute(E, E_prev, R, m, n);
#endif
    // We only print the meshes if they are small enough
    // #if 1
    //     printMat("E_prev",E_prev,m,n);
    //     printMat("R",R,m,n);
    // #endif
}

void distribute(double *E,double *E_prev,double *R,int m,int n) {
    int px = cb.px, py = cb.py;
    int lowMatM = m / py, lowMatN = n / px, extraM = m % py, extraN = n % px;
    int nprocs, curRank;
    MPI_Comm_rank(MPI_COMM_WORLD,&curRank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    int base = 0;
    if (curRank == 0) {
        // printMat("E_prev", E_prev, m, n);
        // send data
        for (int iterRank = 1; iterRank < nprocs; iterRank++) {
            int rankRowIdx = iterRank / px, rankColIdx = iterRank % px, subMatM = lowMatM, subMatN = lowMatN;
            // where the first element located
            // 4
            // extraM = 0 extraN = 2 subMatM = 4 subMatN = 2 px = 2 py = 3
            // rankRowIdx = 1 < 2 -> 
            int iterRowIdx, iterColIdx;
            if (rankRowIdx < (py - extraM)) {
                iterRowIdx = subMatM * rankRowIdx;
            } else {
                subMatM++;
                iterRowIdx = subMatM * rankRowIdx - (py - extraM);
            }
            if (rankColIdx < (px - extraN)) {
                iterColIdx = subMatN * rankColIdx;
            } else {
                subMatN++;
                iterColIdx = subMatN * rankColIdx - (px - extraN);
            }
            // PADDING
            subMatN+=2;
            subMatM+=2;
            double* curE = alloc1D(subMatM, subMatN);
            double* curE_Prev = alloc1D(subMatM, subMatN);
            double* curR = alloc1D(subMatM, subMatN);
            for (int i = 0; i < subMatM; i++) {
                for (int j = 0; j < subMatN; j++) {
                    curE[i*subMatN + j] =      E[(iterRowIdx + i)*(n+2) + iterColIdx + j];
                    curE_Prev[i*subMatN + j] = E_prev[(iterRowIdx + i)*(n+2) + iterColIdx + j];
                    curR[i*subMatN + j] =      R[(iterRowIdx + i)*(n+2) + iterColIdx + j];
                }
            }

            // printf("cur Rank %d\n", iterRank);
            // printf("cur row %d\n", iterRowIdx);
            // printf("cur col %d\n", iterColIdx);
            // printMat("E_prev", curE_Prev, subMatM-2, subMatN-2);

            MPI_Send(curE, subMatM*subMatN, MPI_DOUBLE, iterRank, 0, MPI_COMM_WORLD);
            MPI_Send(curE_Prev, subMatM*subMatN, MPI_DOUBLE, iterRank, 1, MPI_COMM_WORLD);
            MPI_Send(curR, subMatM*subMatN, MPI_DOUBLE, iterRank, 2, MPI_COMM_WORLD);
        }
        // process 0 block
        int subMatM = lowMatM + 2, subMatN = lowMatN + 2;
        for (int i = 0; i < subMatM; i++) {
            for (int j = 0; j < subMatN; j++) {
                E[i*subMatN + j] = E[i*(n+2) + j];
                E_prev[i*subMatN + j] = E_prev[i*(n+2) + j];
                R[i*subMatN + j] = R[i*(n+2) + j];
            }
        }
        
    } else {
        int rankRowIdx = curRank / px, rankColIdx = curRank % px;
        int subMatM = rankRowIdx < (py - extraM) ? lowMatM + 2 : lowMatM + 3;
        int subMatN = rankColIdx < (px - extraN) ? lowMatN + 2 : lowMatN + 3;
        MPI_Recv(E, subMatM*subMatN, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(E_prev, subMatM*subMatN, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(R, subMatM*subMatN, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

double *alloc1D(int m,int n){
    int nx=n, ny=m;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E= (double*) memalign(16, sizeof(double)*nx*ny) );
    return(E);
}

void printMat(const char mesg[], double *E, int m, int n){
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m>34)
      return;
#endif
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
       int rowIndex = i / (n+2);
       int colIndex = i % (n+2);
       if ((colIndex>0) && (colIndex<n+1))
          if ((rowIndex > 0) && (rowIndex < m+1))
            printf("%6.3f ", E[i]);
       if (colIndex == n+1)
        printf("\n");
    }
}
