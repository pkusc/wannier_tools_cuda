#pragma once

typedef size_t devptr_t;

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */
int CUBLAS_INIT (void);
int CUBLAS_SHUTDOWN (void);
int CUBLAS_ALLOC (const int *n, const int *elemSize, devptr_t *devicePtr);
int CUBLAS_FREE (const devptr_t *devicePtr);
int CUBLAS_SET_VECTOR (const int *n, const int *elemSize, const void *x,
                       const int *incx, const devptr_t *y, const int *incy);
int CUBLAS_GET_VECTOR (const int *n, const int *elemSize, const devptr_t *x,
                       const int *incx, void *y, const int *incy);
int CUBLAS_SET_MATRIX (const int *rows, const int *cols, const int *elemSize,
                       const void *A, const int *lda, const devptr_t *B, 
                       const int *ldb);
int CUBLAS_GET_MATRIX (const int *rows, const int *cols, const int *elemSize,
                       const devptr_t *A, const int *lda, void *B, const int *ldb);

/* BLAS util */
void CUBLAS_XERBLA (const char *srName, int *info);
int CUBLAS_GET_ERROR (void);

#if defined(__cplusplus)
}
#endif /* __cplusplus */


/*
 * Fortran callable thin wrappers. Fortran application must allocate and
 * deallocate GPU memory, and copy data up and down.
 */
#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

void CUBLAS_ZGEMM (const char *transa, const char *transb, const int *m,
                   const int *n, const int *k, const cuDoubleComplex *alpha,
                   const devptr_t *devPtrA, const int *lda,
                   const devptr_t *devPtrB, const int *ldb, 
                   const cuDoubleComplex *beta, const devptr_t *devPtrC,
                   const int *ldc);                                                                    

#if defined(__cplusplus)
}
#endif /* __cplusplus */