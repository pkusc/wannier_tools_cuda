#include "cublas.h"   /* CUBLAS public header file  */

#include "cumat_mul_common.h"
#include "cumat_mul.h"

#include "cuda_set_device.h"

void cuda_set_device();
void cuda_my_memset(void* ptr, int value, size_t count);

int CUBLAS_INIT (void) 
{
    cuda_set_device();
    return (int)cublasInit ();
}

int CUBLAS_SHUTDOWN (void) 
{
    return (int)cublasShutdown ();
}

int CUBLAS_ALLOC (const int *n, const int *elemSize, devptr_t *devicePtr)
{    
    void *tPtr;
    int retVal;
    retVal = (int)cublasAlloc (*n, *elemSize, &tPtr);
    *devicePtr = (devptr_t)tPtr;
    return retVal;
}

int CUBLAS_FREE (const devptr_t *devicePtr)
{
    void *tPtr;
    tPtr = (void *)(*devicePtr);
    return (int)cublasFree (tPtr);
}

int CUBLAS_SET_VECTOR (const int *n, const int *elemSize, const void *x,
                       const int *incx, const devptr_t *y, const int *incy)
{
    void *tPtr = (void *)(*y);
    return (int)cublasSetVector (*n, *elemSize, x, *incx, tPtr, *incy);
}

int CUBLAS_GET_VECTOR (const int *n, const int *elemSize, const devptr_t *x,
                       const int *incx, void *y, const int *incy)
{
    const void *tPtr = (const void *)(*x);
    return (int)cublasGetVector (*n, *elemSize, tPtr, *incx, y, *incy);
}

int CUBLAS_SET_MATRIX (const int *rows, const int *cols, const int *elemSize,
                       const void *A, const int *lda, const devptr_t *B, 
                       const int *ldb)
{
    void *tPtr = (void *)(*B);
    return (int)cublasSetMatrix (*rows, *cols, *elemSize, A, *lda, tPtr,*ldb);
}

int CUBLAS_GET_MATRIX (const int *rows, const int *cols, const int *elemSize,
                       const devptr_t *A, const int *lda, void *B, 
                       const int *ldb)
{
    const void *tPtr = (const void *)(*A);
    return (int)cublasGetMatrix (*rows, *cols, *elemSize, tPtr, *lda, B, *ldb);
}

int CUBLAS_GET_ERROR (void)
{
    return (int)cublasGetError();
}

void CUBLAS_XERBLA (const char *srName, int *info)
{
    cublasXerbla (srName, *info);
}



void CUBLAS_ZGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const cuDoubleComplex *alpha,
                   const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrB, const int *ldb, 
                   const cuDoubleComplex *beta, const devptr_t *devPtrC,
                   const int *ldc)
{
    cuDoubleComplex *A = (cuDoubleComplex *)*devPtrA;
    cuDoubleComplex *B = (cuDoubleComplex *)*devPtrB;
    cuDoubleComplex *C = (cuDoubleComplex *)*devPtrC;    
    cublasZgemm (transa[0], transb[0], *m, *n, *k, *alpha, A, *lda, B, *ldb, 
                 *beta, C, *ldc);
}

// cublas_set_memset(nmatdim*nmatdim,sizeOf(C(1,1)),devPtrC)
void CUBLAS_MEMSET (const int* n, const int* sz, const devptr_t* devPtr)
{
    void *tPtr = (void *)(*devPtr);
    cuda_my_memset(tPtr, 0, *n * *sz);
}