/* 
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 * 
 * Modified and  restructured by Scott B. Baden, UCSD
 * 
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>
#include <mpi.h>
using namespace std;

double *alloc1D(int m,int n);
void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
void solveGivenSingleProcess(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf);
void solveWithMPI(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf);
extern control_block cb;

// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif 

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
	double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
	l2norm = sqrt(l2norm);
	return l2norm;
}


void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf) {
#ifdef _MPI_
	solveWithMPI(_E, _E_prev, R, alpha, dt, plotter, L2, Linf);
#else
	solveGivenSingleProcess(_E, _E_prev, R, alpha, dt, plotter, L2, Linf);
#endif
}


void solveWithMPI(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf) {
	// Simulated time is different from the integer timestep number
	double t = 0.0;
	double *E = *_E, *E_prev = *_E_prev;
	double *R_tmp = R;
	double *E_tmp = *_E;
	double *E_prev_tmp = *_E_prev;
	double mx, sumSq;
	int niter;
	int m = cb.m, n=cb.n, px = cb.px, py = cb.py;
	int nprocs, curRank, i, j, extraM = m % py, extraN = n % px;

	// number of process should less than matrix size

	//evenly distribute work load
	MPI_Comm_rank(MPI_COMM_WORLD,&curRank);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

	int curRowIdx = curRank / px; 
	int curColIdx = curRank % px;
	int subMatM = m / py + 2, subMatN = n / px + 2;

    subMatM += curRowIdx >= (py - extraM);
    subMatN += curColIdx >= (px - extraN);

    //printMat2("E_prev", E_prev, subMatM-2, subMatN-2);
	double* leftPackSend = alloc1D(subMatM, 1);
	double* leftPackRecv = alloc1D(subMatM, 1);
	double* rightPackSend = alloc1D(subMatM, 1);
	double* rightPackRecv = alloc1D(subMatM, 1);
	// MPI_Irecv(buf, count, type, src, tag, comm, &request)
	// MPI_Isend(buf, count, type, dest, tag, comm, &request)
	MPI_Request r10, r20, r30, r40, r11, r21, r31, r41;
	MPI_Status s10, s20, s30, s40, s11, s21, s31, s41;

	for (niter = 0; niter < cb.niters; niter++) {
		if  (cb.debug && (niter==0)){
			stats(E_prev,m,n,&mx,&sumSq);
			double l2norm = L2Norm(sumSq);
			repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
			if (cb.plot_freq)
				plotter->updatePlot(E,  -1, m+1, n+1);
		} 
		if (!cb.noComm){
		if (curRowIdx > 0) {
			MPI_Isend(E_prev + subMatN, subMatN, MPI_DOUBLE, curRank - px, 0, MPI_COMM_WORLD, &r10);
			MPI_Irecv(E_prev, subMatN, MPI_DOUBLE, curRank - px, 0, MPI_COMM_WORLD, &r11);
		} else {
			for (int i = 0; i < subMatN; i++) {
				E_prev[i] = E_prev[i + 2 * subMatN];
			}
		}

		if (curRowIdx < py - 1) {
			MPI_Isend(E_prev + subMatN * (subMatM-2), subMatN, MPI_DOUBLE, curRank + px, 0, MPI_COMM_WORLD, &r20);
			MPI_Irecv(E_prev + subMatN * (subMatM-1), subMatN, MPI_DOUBLE, curRank + px, 0, MPI_COMM_WORLD, &r21);
		} else {
			int base1 = subMatN * (subMatM-1);
			int base2 = subMatN * (subMatM-3);
			for (int i = 0; i < subMatN; i++) {
				E_prev[base1 + i] = E_prev[base2 + i];
			}
		}

		if (curColIdx > 0) {
			for (int i = 0; i < subMatM; i++) {
				leftPackSend[i] = E_prev[i*subMatN + 1];
			}
			MPI_Isend(leftPackSend, subMatM, MPI_DOUBLE, curRank - 1, 0, MPI_COMM_WORLD, &r30);
			MPI_Irecv(leftPackRecv, subMatM, MPI_DOUBLE, curRank - 1, 0, MPI_COMM_WORLD, &r31);
		} else {
			for (int i = 0; i < subMatM; i++) {
				E_prev[i*subMatN] = E_prev[i*subMatN + 2];
			}
		}

		if (curColIdx < px - 1) {
			for (int i = 1; i <= subMatM; i++) {
				rightPackSend[i-1] = E_prev[i*subMatN - 2];
			}
			MPI_Isend(rightPackSend, subMatM, MPI_DOUBLE, curRank + 1, 0, MPI_COMM_WORLD, &r40);
			MPI_Irecv(rightPackRecv, subMatM, MPI_DOUBLE, curRank + 1, 0, MPI_COMM_WORLD, &r41);
		} else {
			for (int i = 1; i <= subMatM; i++) {
				E_prev[i * subMatN-1] = E_prev[i * subMatN-3];
			}
		}
		}

//////////////////////////////////////////////////////////////////////////////
		int innerBlockRowStartIndex = 2 * subMatN + 1, innerBlockRowEndIndex = subMatN * (subMatM - 3) + 1;
		int j = 0;
#define FUSED 1
#ifdef FUSED
		// Solve for the excitation, a PDE
		for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=subMatN) {
			E_tmp = E + j;
			E_prev_tmp = E_prev + j;
			R_tmp = R + j;
			register double ePrevTmpInRegister, rTmpInRegister;
			for(i = 1; i < subMatN - 3; i++) {
				ePrevTmpInRegister = E_prev_tmp[i];
				rTmpInRegister = R_tmp[i];
				E_tmp[i] = ePrevTmpInRegister + alpha * (E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*ePrevTmpInRegister+E_prev_tmp[i+subMatN]+E_prev_tmp[i-subMatN]);
				E_tmp[i] += -dt*(kk* ePrevTmpInRegister *(ePrevTmpInRegister-a)*(ePrevTmpInRegister-1)+ePrevTmpInRegister*rTmpInRegister);
				R_tmp[i] += dt*(epsilon+M1* rTmpInRegister/( ePrevTmpInRegister+M2))*(-rTmpInRegister-kk* ePrevTmpInRegister *(ePrevTmpInRegister-b-1));
			}
		}
#else
		// Solve for the excitation, a PDE
		for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=subMatN) {
			E_tmp = E + j;
			E_prev_tmp = E_prev + j;
			for(i = 1; i < subMatN - 3; i++) {
				E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+subMatN]+E_prev_tmp[i-subMatN]);
			}
		}
		/* 
		 * Solve the ODE, advancing excitation and recovery variables
		 *     to the next timtestep
		 */
		for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=subMatN) {
			E_tmp = E + j;
			R_tmp = R + j;
			E_prev_tmp = E_prev + j;
			for(i = 1; i < subMatN - 3; i++) {
				E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
				R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
			}
		}
#endif
		if (!cb.noComm){
		if (curRowIdx > 0) {
			MPI_Wait(&r10, &s10);
			MPI_Wait(&r11, &s11);
		} 

		if (curRowIdx < py - 1) {
			MPI_Wait(&r20, &s20);
			MPI_Wait(&r21, &s21);
		}

		if (curColIdx > 0) {
			MPI_Wait(&r30, &s30);
			MPI_Wait(&r31, &s31);
			for (int i = 0; i < subMatM; i++) {
				E_prev[i * subMatN] = leftPackRecv[i];
			}
		} 

		if (curColIdx < px - 1) {
			MPI_Wait(&r40, &s40);
			MPI_Wait(&r41, &s41);
			for (int i = 1; i <= subMatM; i++) {
				E_prev[i * subMatN-1] = rightPackRecv[i-1];
			}
		}
		}
		
		 /////////////////////////////////////////////////////////////////////////////////
#ifdef FUSED
		E_tmp = E + (subMatN + 1);
		E_prev_tmp = E_prev + (subMatN + 1);
		R_tmp = R + (subMatN + 1);
		register double ePrevTmpInRegister, rTmpInRegister;
		for(i = 1; i < subMatN - 3; i++) {
			ePrevTmpInRegister = E_prev_tmp[i];
			rTmpInRegister = R_tmp[i];
			E_tmp[i] = ePrevTmpInRegister + alpha * (E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*ePrevTmpInRegister+E_prev_tmp[i+subMatN]+E_prev_tmp[i-subMatN]);
			E_tmp[i] += -dt*(kk* ePrevTmpInRegister *(ePrevTmpInRegister-a)*(ePrevTmpInRegister-1)+ePrevTmpInRegister*rTmpInRegister);
			R_tmp[i] += dt*(epsilon+M1* rTmpInRegister/( ePrevTmpInRegister+M2))*(-rTmpInRegister-kk* ePrevTmpInRegister *(ePrevTmpInRegister-b-1));
		}
		E_tmp = E + (subMatN * (subMatM - 2) + 1);
		E_prev_tmp = E_prev + (subMatN * (subMatM - 2) + 1);
		R_tmp = R + (subMatN * (subMatM - 2) + 1);
		for(i = 1; i < subMatN - 3; i++) {
			ePrevTmpInRegister = E_prev_tmp[i];
			rTmpInRegister = R_tmp[i];
			E_tmp[i] = ePrevTmpInRegister + alpha * (E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*ePrevTmpInRegister+E_prev_tmp[i+subMatN]+E_prev_tmp[i-subMatN]);
			E_tmp[i] += -dt*(kk* ePrevTmpInRegister *(ePrevTmpInRegister-a)*(ePrevTmpInRegister-1)+ePrevTmpInRegister*rTmpInRegister);
			R_tmp[i] += dt*(epsilon+M1* rTmpInRegister/( ePrevTmpInRegister+M2))*(-rTmpInRegister-kk* ePrevTmpInRegister *(ePrevTmpInRegister-b-1));
		}
		innerBlockRowStartIndex = subMatN + 1;
		innerBlockRowEndIndex = subMatN * (subMatM - 2) + 1;
		for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=subMatN) {
			E_tmp = E + j;
			E_prev_tmp = E_prev + j;
			R_tmp = R + j;
			i = 0;
			ePrevTmpInRegister = E_prev_tmp[i];
			rTmpInRegister = R_tmp[i];
			E_tmp[i] = ePrevTmpInRegister + alpha * (E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*ePrevTmpInRegister+E_prev_tmp[i+subMatN]+E_prev_tmp[i-subMatN]);
			E_tmp[i] += -dt*(kk* ePrevTmpInRegister *(ePrevTmpInRegister-a)*(ePrevTmpInRegister-1)+ePrevTmpInRegister*rTmpInRegister);
			R_tmp[i] += dt*(epsilon+M1* rTmpInRegister/( ePrevTmpInRegister+M2))*(-rTmpInRegister-kk* ePrevTmpInRegister *(ePrevTmpInRegister-b-1));
			i = subMatN - 3;
			ePrevTmpInRegister = E_prev_tmp[i];
			rTmpInRegister = R_tmp[i];
			E_tmp[i] = ePrevTmpInRegister + alpha * (E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*ePrevTmpInRegister+E_prev_tmp[i+subMatN]+E_prev_tmp[i-subMatN]);
			E_tmp[i] += -dt*(kk* ePrevTmpInRegister *(ePrevTmpInRegister-a)*(ePrevTmpInRegister-1)+ePrevTmpInRegister*rTmpInRegister);
			R_tmp[i] += dt*(epsilon+M1* rTmpInRegister/( ePrevTmpInRegister+M2))*(-rTmpInRegister-kk* ePrevTmpInRegister *(ePrevTmpInRegister-b-1));			
		}
#endif

		if (cb.stats_freq){
			 if ( !(niter % cb.stats_freq)){
				stats(E,subMatM-2,subMatN-2,&mx,&sumSq);
				double l2norm = L2Norm(sumSq);
				repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
			}
		}

		if (cb.plot_freq){
			if (!(niter % cb.plot_freq)){
				plotter->updatePlot(E,  niter, m, n);
			}
		}

		// Swap current and previous meshes
		double *tmp = E; E = E_prev; E_prev = tmp;

	}

	stats(E_prev, subMatM - 2, subMatN - 2, &Linf, &sumSq);
    double curLinf, curSumSq;
    MPI_Reduce(&Linf, &curLinf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sumSq, &curSumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    Linf = curLinf;
    sumSq = curSumSq;
	L2 = L2Norm(sumSq);
	// Swap pointers so we can re-use the arrays
	*_E = E;
	*_E_prev = E_prev;

}

void solveGivenSingleProcess(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

 // Simulated time is different from the integer timestep number
 double t = 0.0;

 double *E = *_E, *E_prev = *_E_prev;
 double *R_tmp = R;
 double *E_tmp = *_E;
 double *E_prev_tmp = *_E_prev;
 double mx, sumSq;
 int niter;
 int m = cb.m, n=cb.n;
 int innerBlockRowStartIndex = (n+2)+1;
 int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);


 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
	for (niter = 0; niter < cb.niters; niter++){
	
		if  (cb.debug && (niter==0)){
			stats(E_prev,m,n,&mx,&sumSq);
			double l2norm = L2Norm(sumSq);
			repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
			if (cb.plot_freq)
				plotter->updatePlot(E,  -1, m+1, n+1);
		}

	 /* 
		* Copy data from boundary of the computational box to the
		* padding region, set up for differencing computational box's boundary
		*
		* These are physical boundary conditions, and are not to be confused
		* with ghost cells that we would use in an MPI implementation
		*
		* The reason why we copy boundary conditions is to avoid
		* computing single sided differences at the boundaries
		* which increase the running time of solve()
		*
		*/
		
		// 4 FOR LOOPS set up the padding needed for the boundary conditions
		int i,j;

		// Fills in the TOP Ghost Cells
		for (i = 0; i < (n+2); i++) {
			E_prev[i] = E_prev[i + (n+2)*2];
		}

		// Fills in the RIGHT Ghost Cells
		for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) {
			E_prev[i] = E_prev[i-2];
		}

		// Fills in the LEFT Ghost Cells
		for (i = 0; i < (m+2)*(n+2); i+=(n+2)) {
			E_prev[i] = E_prev[i+2];
		}	

		// Fills in the BOTTOM Ghost Cells
		for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) {
			E_prev[i] = E_prev[i - (n+2)*2];
		}

//////////////////////////////////////////////////////////////////////////////

#define FUSED 1

#ifdef FUSED
		// Solve for the excitation, a PDE
		for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
			E_tmp = E + j;
			E_prev_tmp = E_prev + j;
			R_tmp = R + j;
			for(i = 0; i < n; i++) {
				E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
				E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
				R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
			}
		}
#else
		// Solve for the excitation, a PDE
		for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
			E_tmp = E + j;
			E_prev_tmp = E_prev + j;
			for(i = 0; i < n; i++) {
				E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
			}
		}

		/* 
		 * Solve the ODE, advancing excitation and recovery variables
		 *     to the next timtestep
		 */

		for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
			E_tmp = E + j;
			R_tmp = R + j;
			E_prev_tmp = E_prev + j;
			for(i = 0; i < n; i++) {
				E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
				R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
			}
		}
#endif
		 /////////////////////////////////////////////////////////////////////////////////

		if (cb.stats_freq){
			 if ( !(niter % cb.stats_freq)){
				stats(E,m,n,&mx,&sumSq);
				double l2norm = L2Norm(sumSq);
				repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
			}
		}

		if (cb.plot_freq){
			if (!(niter % cb.plot_freq)){
				plotter->updatePlot(E,  niter, m, n);
			}
		}

		// Swap current and previous meshes
		double *tmp = E; E = E_prev; E_prev = tmp;

 	} //end of 'niter' loop at the beginning

	//  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

	stats(E_prev,m,n,&Linf,&sumSq);
	L2 = L2Norm(sumSq);

	// Swap pointers so we can re-use the arrays
	*_E = E;
	*_E_prev = E_prev;
}

void printMat2(const char mesg[], double *E, int m, int n){
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

