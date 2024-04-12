module matmuldata
   implicit none
   integer(8) :: devPtrA, devPtrB, devPtrC
   integer :: last_nmatdim = 0
end module matmuldata

! performs matrix-matrix multiply
! C=A*B

  subroutine mat_mul(nmatdim,A,B,C,TestInit)
     
     use para, only : Dp
     use matmuldata, only : devPtrA, devPtrB, devPtrC, last_nmatdim
      implicit none

     integer,intent(in) :: nmatdim  
     integer, pointer, optional :: TestInit

     complex(Dp) :: ALPHA
     complex(Dp) :: BETA 
 
     complex(Dp), intent(in)  :: A(nmatdim ,nmatdim)
     complex(Dp), intent(in)  :: B(nmatdim ,nmatdim)
     complex(Dp), intent(out) :: C(nmatdim,nmatdim)

     C(:,:)=(0.0d0,0.0d0)

     ! print *, 'Size of nmatdim is: ', nmatdim

      if (last_nmatdim .eq. 0) then
         call cublas_alloc(nmatdim*nmatdim, sizeOf(A(1,1)), devPtrA)
         call cublas_alloc(nmatdim*nmatdim, sizeOf(B(1,1)), devPtrB)
         call cublas_alloc(nmatdim*nmatdim, sizeOf(C(1,1)), devPtrC)
         last_nmatdim = nmatdim
      endif


      ! If last_nmatdim less than nmatdim, we need to free the old memory and allocate new memory
      if (last_nmatdim .lt. nmatdim) then
            call cublas_free(devPtrA)
            call cublas_free(devPtrB)
            call cublas_free(devPtrC)
            call cublas_alloc(nmatdim*nmatdim, sizeOf(A(1,1)), devPtrA)
            call cublas_alloc(nmatdim*nmatdim, sizeOf(B(1,1)), devPtrB)
            call cublas_alloc(nmatdim*nmatdim, sizeOf(C(1,1)), devPtrC)
            last_nmatdim = nmatdim
      endif

     call cublas_set_matrix(nmatdim,nmatdim,sizeOf(A(1,1)),A,nmatdim,devPtrA,nmatdim)
     call cublas_set_matrix(nmatdim,nmatdim,sizeOf(B(1,1)),B,nmatdim,devPtrB,nmatdim)
     call cublas_memset(nmatdim*nmatdim,sizeOf(C(1,1)),devPtrC)

     ALPHA=1.0d0 
     BETA=0.0D0

     call cublas_zgemm('n','n',nmatdim,nmatdim,nmatdim,ALPHA,devPtrA,nmatdim,devPtrB,nmatdim,BETA,devPtrC,nmatdim)

     call cublas_get_matrix(nmatdim,nmatdim,sizeOf(C(1,1)),devPtrC,nmatdim,C,nmatdim)
     return
  end subroutine mat_mul

  !> ZGESVD computes the singular value decomposition (SVD) for GE matrices
  !> In this pack, we assume the matrix A is a square matrix, the dimension 
  !> of row and column are the same
  !> A = U * SIGMA * conjugate-transpose(V)
  !> VT= conjugate-transpose(V)
  subroutine zgesvd_pack(M, A, U, S, VT)

     use para, only : Dp
     implicit none

     integer, intent(in) :: M
     complex(dp), intent(inout) :: A(M, M)
     complex(dp), intent(out) :: U(M, M)
     real(dp)   , intent(out) :: S(M, M)
     complex(dp), intent(out) :: VT(M, M)

     character :: JOBU
     character :: JOBVT
     integer :: N
     integer :: LDA
     integer :: LDU
     integer :: LDVT
     integer :: LWORK
     real(dp), allocatable :: RWORK(:)
     complex(dp), allocatable :: WORK(:)
     integer :: INFO

     N= M
     LDA= M
     LDU= M
     LDVT= M
     allocate(RWORK(5*M))

     allocate(work(5*M))

     JOBU= 'A'
     JOBVT= 'A'

     LWORK = -1
     call zgesvd (JOBU, JOBVT, M, N, A, LDA, S, U, LDU, &
        VT, LDVT, WORK, LWORK, RWORK, INFO)
     if (INFO==0 .and. real(WORK(1))>0 )then
        LWORK= WORK(1)
        deallocate(work)
        allocate(WORK(LWORK))
     else
        write(*, *)'something wrong with zgesvd'
     endif


     call zgesvd (JOBU, JOBVT, M, N, A, LDA, S, U, LDU, &
        VT, LDVT, WORK, LWORK, RWORK, INFO)
     if (INFO /= 0) write(*, *)'something wrong with zgesvd'

     return
  end subroutine zgesvd_pack

  subroutine zhpevx_pack(mat,ndim,eig,rot)
    !                                                            !
    ! Diagonalize the ndim x ndim  hermitian matrix 'mat' and      !
    ! return the eigenvalues 'eig' and the unitary rotation 'rot'!
    !                                                            !
    !============================================================!

    use para, only : dp, stdout

    integer, intent(in)           :: ndim
    complex(dp), intent(in)  :: mat(ndim,ndim)
    real(dp), intent(out)    :: eig(ndim)
    complex(dp), intent(out) :: rot(ndim,ndim)

    complex(dp), allocatable :: mat_pack(:),cwork(:)
    real(dp), allocatable    :: rwork(:)
    integer            :: i,j,info,nfound
    integer, allocatable :: iwork(:),ifail(:)

    allocate(mat_pack((ndim*(ndim+1))/2))
    allocate(cwork(2*ndim))
    allocate(rwork(7*ndim))
    allocate(iwork(5*ndim))
    allocate(ifail(ndim))
    do j=1,ndim
       do i=1,j
          mat_pack(i+((j-1)*j)/2)=mat(i,j)
       enddo
    enddo
    rot=0d0;eig=0.0_dp;cwork=0d0;rwork=0.0_dp;iwork=0
    call ZHPEVX('V','A','U',ndim,mat_pack,0.0_dp,0.0_dp,0,0,-1.0_dp, &
         nfound,eig(1),rot,ndim,cwork,rwork,iwork,ifail,info)
    if(info < 0) then
       write(stdout,'(a,i3,a)') 'THE ',-info,&
            ' ARGUMENT OF ZHPEVX HAD AN ILLEGAL VALUE'
       stop 'Error in zhpevx_pack'
    endif
    if(info > 0) then
       write(stdout,'(i3,a)') info,' EIGENVECTORS FAILED TO CONVERGE'
       stop 'Error in zhpevx_pack'
    endif

    return
  end subroutine zhpevx_pack

 
