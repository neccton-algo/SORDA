# Operating System (Linux (default),AIX or IRIX)
# Mac OS X 

OS ?= Linux

# Compiler (gfortran (default), ifort, pgi, g95)

FORT ?= gfortran

# Binary format (big_endian (default) or little_endian)
# This setting is not used if you only use NetCDF files

FORMAT ?= big_endian

# Precision (single or double (default))

PRECISION ?= double

# Suffix for executables

EXEC_SUFFIX ?= 

# Install directory for OAK (current working directory is the default) 

OAK_DIR ?= $(CURDIR)

# uncomment/comment this line to activate/deactivate MPI support
MPI ?= on

# uncomment/comment this line to activate/deactivate compilation with mpif90 
# Otherwise, the Fortran compiler is directly used the variables 
# MPI_INCDIR, MPI_LIBDIR and MPI_LIB.
# The flag is only used if MPI is activated.

USE_MPIF90 ?= on

# uncomment/comment this line to activate/deactivate OpenMP support

#OPENMP ?= on


# uncomment/comment this line to activate/deactivate debugger information

#DEBUG ?= on

# ------------------------------------------------------------------------------------------------------------
# Library names and locations
# Variables name use the following naming scheme:
# 
# <library>_INCDIR: directory containing the include files and compiled modules (e.g. /home/user/mpi/include)
# <library>_LIBDIR: directory containing the library  (e.g. /home/user/mpi/lib)
# <library>_LIB: name of the library (including dependencies) for the linker (e.g. -lmpi -lpthread)
#

# netCDF library
# Note if the nc-config tool is present and NETCDF_CONFIG is defined, then the values of 
# NETCDF_INCDIR, NETCDF_LIBDIR and NETCDF_LIB are not used

#NETCDF_CONFIG ?= nc-config
#NETCDF_INCDIR ?= 
#NETCDF_LIBDIR ?= 
#NETCDF_LIB ?= 

# MPI library
# Only used if USE_MPIF90 is deactivated
#MPI_INCDIR ?= 
#MPI_LIBDIR ?= 
#MPI_LIB ?= 

# Lapack library

LAPACK_LIBDIR ?= /opt/cecisw/arch/easybuild/2023b/software/ScaLAPACK/2.2.0-gompi-2023b-fb/lib
LAPACK_LIB ?= -lscalapack

# BLAS library

#BLAS_LIBDIR ?=
#BLAS_LIB ?= -lblas
#BLAS_LIBDIR=$MKLROOT/lib/intel64/    # for ifort on nic5
#BLAS_LIB=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core  # for ifort on nic5
BLAS_LIBDIR = /opt/cecisw/arch/easybuild/2023b/software/OpenBLAS/0.3.24-GCC-13.2.0/lib
BLAS_INCDIR = /opt/cecisw/arch/easybuild/2023b/software/OpenBLAS/0.3.24-GCC-13.2.0/include
BLAS_LIB= -lopenblas

# additional compiler and linker options (such as libraries)
#EXTRA_F90FLAGS ?= 
#EXTRA_LDFLAGS ?= 

CHOLMOD_LIB =
