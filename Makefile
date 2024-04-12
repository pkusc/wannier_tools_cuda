OBJ =  module.o sparse.o wt_aux.o math_lib.o symmetry.o readHmnR.o inverse.o proteus.o \
       eigen.o ham_qlayer2qlayer.o psi.o unfolding.o rand.o \
  		 ham_slab.o ham_bulk.o ek_slab.o ek_bulk_polar.o ek_bulk.o \
       readinput.o fermisurface.o surfgreen.o surfstat.o \
  		 mat_mul.o ham_ribbon.o ek_ribbon.o \
       fermiarc.o berrycurvature.o \
  		 wanniercenter.o dos.o  orbital_momenta.o \
  		 landau_level_sparse.o landau_level.o lanczos_sparse.o \
		 berry.o wanniercenter_adaptive.o \
  		 effective_mass.o findnodes.o \
		 sigma_OHE.o sigma.o Boltz_transport_anomalous.o \
	main.o

# compiler
F90  = mpiifort -fpp -DMPI -DINTELMKL -fpe3

INCLUDE = -I${MKLROOT}/include
WFLAG = -nogen-interface
OFLAG = -O3 -static-intel -xHost -unroll=8
FFLAG = $(OFLAG) $(WFLAG)
LFLAG = $(OFLAG)

# ARPACK LIBRARY
ARPACK=/data/wzf/mis/libarpack.a

# blas and lapack libraries
# static linking
LIBS =  -L/usr/local/cuda/lib64 -lcublas -lcudart ${ARPACK} -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a \
        ${MKLROOT}/lib/intel64/libmkl_sequential.a \
        ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl \
		-L/opt/intel/oneapi/mpi/latest/lib -lmpi

# dynamic linking
# LIBS = ${ARPACK} -L/${MKLROOT}/lib/intel64 -lmkl_core -lmkl_sequential -lmkl_intel_lp64 -lpthread


main : $(OBJ) cu_mat_mul.o cuda_set_device.o
	$(F90) $(LFLAG) $(OBJ) -o wt.x $(LIBS) cu_mat_mul.o cuda_set_device.o -lcublas
	cp -f wt.x ../bin

cu_mat_mul.o : cumat_mul.c cuda_set_device.o
	mpiicx -c cumat_mul.c -o cu_mat_mul.o -I/usr/local/cuda/include -I/opt/intel/oneapi/mpi/latest/include 

cuda_set_device.o : cuda_set_device.cu
	nvcc -O3 -gencode=arch=compute_80,code=sm_80 -c cuda_set_device.cu -o cuda_set_device.o -I/opt/intel/oneapi/mpi/latest/include

.SUFFIXES: .o .f90

.f90.o :
	$(F90) $(FFLAG) $(INCLUDE) -c $*.f90

clean :
	rm -f *.o *.mod *~ wt.x

