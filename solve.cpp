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

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
double *alloc1D(int m,int n);

#define TAG_TOP 0
#define TAG_BOTTOM 1
#define TAG_LEFT 2
#define TAG_RIGHT 3


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

void copy_mat(double *m1, double *m2, int stride, int n){
    for (i=0;i<n;i+=stride){
        m2[i*stride]=m1[i*stride];
    }
}

void stats_mpi(double *E, int cm, int cn, int stride, double *_mx, double *sumSq){
     double mx = -1;
     double _sumSq = 0;
     for (int i=0;i<cm;i++){
         for(int j=0;j<cn;j++){
             _index=(i+1)*stride+(j+1);
             sumSq+=E[_index]*E[_index];
             double fe= fabs(E[i]);
             if(fe>mx){
                 mx=fe;
             }
         }
     }
     
     MPI_Reduce(&mx,&mx,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
     MPI_Reduce(&sumSq,&sumSq,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
     *_mx=mx;
     *_sumSq=sumSq;
}

void prepare_scatter_matrix(double *m1, double *m2,int px, int py, int x_b, int y_b, int bm, int bn){
    int m1_offset,m2_offset;
    for (int idx=0;idx<px;idx++){
        for (int idy=0;idy<py;idy++){
            m2_offset=((idx*py)+idy)*bm*bn;
            int addx = (idx < x_b)? 0 : (idx - x_b);
            int addy = (idy < y_b)? 0 : (idy - y_b);
            int prevx = (idx < x_b) ? idx : x_b;
            int prevy = (idy < y_b) ? idy : y_b;
            int cm = idx < x_b ? bm : (bm - 1);
            int cn = idy < y_b ? bn : (bn - 1);
            int offset = (prevx * bm + addx * (bm - 1) + 1) * stride + (prevy * bn + addy * (bn - 1)) + 1;
            for (int i=0;i<cm;i++){
                for (int j=0;j<cn;j++){
                    m2[m2_offset+i*bn+j]=m1[m1_offset+i*stride+j];
                }
            }
        }
    }
}


void scatter_mpi(double *E, double *E_curr, double *R, double *R_curr, int rank, int px, int py, int x_b, int y_b,int bm, int bn){
    double *E_scatter=alloc1D(px*bm,py*bn);
    double *R_scatter=alloc1D(px*bm,py*bn);
    double *E_recv_scatter=alloc1D(bm,bn);
    double *R_recv_scatter=alloc1D(bm,bn);
    if (rank==0){
        prepare_scatter_matrix(E,E_scatter,px,py,x_b,y_b,bm,bn);
        prepare_scatter_matrix(R,R_scatter,px,py,x_b,y_b,bm,bn);
    }
    MPI_Scatter(E_scatter,bm*bn,MPI_DOUBLE,E_recv_scatter,bm*bn,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Scatter(R_scatter,bm*bn,MPI_DOUBLE,R_recv_scatter,bm*bn,MPI_DOUBLE,0,MPI_COMM_WORLD);
    for (int i=0;i<bm;i++){
        for (int j=0;j<bn;j++){
            E_curr[(i+1)*(bn+2)+j+1]=E_recv_scatter[i*bn+j];
            R_curr[(i+1)*(bn+2)+j+1]=R_recv_scatter[i*bn+j];
        }
    }
}

#define RANKID(idx,idy,row_stride) ((x)*(row_stride)+(y))

void Communicate(double *e_prev, int idx, int idy, int px, int py, int bm, int bn, int cm, int cn){
    MPI_Request recv_req[4], send_req[4];
    MPI_Status recv_status[4], send_status[4];
    stride=bn+2
    MPI_Datatype col_data;
    MPI_Type_vector(cm,1,stride,MPI_DOUBLE,&col_data);
    MPI_Type_commit(&col_data);
    
    if (idx==0){
        copy_mat(e_prev+2*stride+1,e_prev+1,1,cn);
    }
    else{
        MPI_Isend(e_prev+stride+1,cn,MPI_DOUBLE,RANKID(idx-1,idy,py),TAG_TOP,MPI_COMM_WORLD,&send_req[0]);
        MPI_Irecv(e_prev+1,cn,MPI_DOUBLE,RANKID(idx-1,idy,py),TAG_BOTTOM,MPI_COMM_WORLD,&recv_req[0]);
    }

    if (idx==px-1){
        copy_mat(e_prev+(cm-1)*stride+1,e_prev+(cm+1)*stride+1,1,cn);
    }
    else{
        MPI_Isend(e_prev+cm*stride+1,cn,MPI_DOUBLE,RANKID(idx+1,idy),TAG_BOTTOM,MPI_COMM_WORLD,&send_req[1]);
        MPI_Irecv(e_prev+(cm+1)*stride+1,cn,MPI_DOUBLE,RANKID(idx+1,idy),TAG_TOP,MPI_COMM_WORLD,&recv_req[1]);
    }

    if (idy==0){
        copy_mat(e_prev+2+stride,e_prev+stride,stride,cm);
    }
    else{
        MPI_Isend(e_prev+1+stride,1,col_data,RANKID(idx,idy-1),TAG_LEFT,MPI_COMM_WORLD,&send_req[2]);
        MPI_Irecv(e_prev+stride,1,col_data,RANKID(idx,idy-1),TAG_RIGHT,MPI_COMM_WORLD,&recv_req[2]);
    }

    if (idy==py-1){
        copy_mat(e_prev+cn-1+stride,e_prev+cn+1+stride,stride,cm);
    }
    else{
        MPI_Isend(e_prev+cn+stride,1,col_data,RANKID(idx,idy+1),TAG_RIGHT,MPI_COMM_WORLD,&send_req[3]);
        MPI_Irecv(e_prev+cn+1+stride,1,col_data,RANKID(idx,idy+1),TAG_LEFT,MPI_COMM_WORLD,&recv_req[3]);
    }

    if(idx>0){
        MPI_WAIT(&send_req[0],&send_status[0]);
        MPI_WAIT(&recv_req[0],&recv_status[0]);
    }
    if (idx<px-1){
        MPI_WAIT(&send_req[1],&send_status[1]);
        MPI_WAIT(&recv_req[1],&recv_status[1]);
    }
    if(idy>0){
        MPI_WAIT(&send_req[2],&send_status[2]);
        MPI_WAIT(&recv_req[2],&recv_status[2]);
    }
    if(idy<py-1){
        MPI_WAIT(&send_req[3],&send_status[3]);
        MPI_WAIT(&recv_req[3],&recv_status[3]);
    }

}

void compute_AP(double *E, double *E_prev, double *R, double alpha, double dt, int stride, int cm, int cn){
    int innerBlockRowStartIndex=stride+1;
    int innerBlockRowEndIndex=cm*stride+1;
#ifdef FUSED
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=stride) {
        double *E_tmp = E + j;
	    double *E_prev_tmp = E_prev + j;
        double *R_tmp = R + j;
        for(int i = 0; i < cn; i++) {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+stride]+E_prev_tmp[i-stride]);
                E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
                R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=stride {
        double *E_tmp = E + j;
        double *E_prev_tmp = E_prev + j;
        for(i = 0; i < cn; i++) {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+stride]+E_prev_tmp[i-stride]);
        }
    }

    /* 
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=stride) {
        double *E_tmp = E + j;
        double *R_tmp = R + j;
	    double *E_prev_tmp = E_prev + j;
        for(i = 0; i < cn; i++) {
	  E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	  R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif
}

void solve_mpi(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){
     double t = 0.0;

    double *E = *_E, *E_prev = *_E_prev;
    double *R_tmp = R;
    double *E_tmp = *_E;
    double *E_prev_tmp = *_E_prev;
    double mx, sumSq;
    int niter;
    int m = cb.m, n=cb.n;
    int px = cb.px, py=cb.py;
    int innerBlockRowStartIndex = (n+2)+1;
    int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    const int idx = rank/py;
    const int idy = rank%py;
    const int bm = (m+px-1)/px;
    const int bn = (n+py-1)/py;
    const int x_b = px-(px*bm-m);
    const int y_b = py-(py*bn-n);

    const int cm = idx<x_b?bm : bm-1;
    const int cn = idy<y_b?bn:bn-1;

    const int stride = bn+2;

    double *e = alloc1D(bm+2,bn+2);
    double *ep=alloc1D(bm+2,bn+2);
    double *r =alloc1D(bm+2,bn+2);


    scatter_mpi(E_prev,ep,R,r,rank,px,py,x_b,y_b,bm,bn);


    for (niter = 0; niter < cb.niters; niter++){
        if  (cb.debug && (niter==0)){
	        stats(E_prev,m,n,&mx,&sumSq);
            double l2norm = L2Norm(sumSq);
	        repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
	        // if (cb.plot_freq)
	        // plotter->updatePlot(E,  -1, m+1, n+1);
        }

        Communicate(ep,idx,idy,px,py,bm,bn,cm,cn);

        compute_AP(e,ep,r,alpha,dt,stride,cm,cn);
        
        if (cb.stats_freq){
            if ( !(niter % cb.stats_freq)){
                stats_mpi(ep,cm,cn,stride,&mx,&sumSq);
                double l2norm = L2Norm(sumSq);
                repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);}}

        // if (cb.plot_freq){
        //     if (!(niter % cb.plot_freq)){
	    //     plotter->updatePlot(E,  niter, m, n);}}

    // Swap current and previous meshes
    double *tmp = e; e = ep; ep = tmp;
    }

    stats_mpi(ep,cm,cn,stride,&Linf,&sumSq);
    // stats(E_prev,m,n,&Linf,&sumSq);
    L2 = L2Norm(sumSq);

    

    // Swap pointers so we can re-use the arrays
    *_E = E;
    *_E_prev = E_prev;

}

void solve_origin(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

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
