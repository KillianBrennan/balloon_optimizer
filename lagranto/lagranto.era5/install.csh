#!/bin/csh

# ---------------------------------------------------------------------- 
# Set some general parameters
# ----------------------------------------------------------------------

# Set LAGRANTO environment variable
setenv LAGRANTO ${PWD}

# Set Fortran compiler
setenv FORTRAN pgf90 

# Init netCDF library depending on the Fortran compiler
if ( "${FORTRAN}" == "pgf90" ) then
  module load netcdf/4.2.1-pgf90
  module list
else if ( "${FORTRAN}" == "gfortran" ) then
  module load gfortran
  module load netcdf/4.2.1
else if ( "${FORTRAN}" == "ifort" ) then
  module load intel
  module load netcdf/4.3.2-intel
else
  echo "Fortran Compiler ${FORTRAN} not supported... Exit"
  exit 1
endif

# Set netCDF paths
setenv NETCDF_LIB `nc-config --flibs`
setenv NETCDF_INC `nc-config --fflags`

# Set list of libraries
set libs  = "iotra ioinp inter times libcdfio libcdfplus"

# Set debug flags
#set fdebug = "-g -O0 -Mbounds -Mchkptr -Mstandard "
set fdebug = ""

# ---------------------------------------------------------------------- 
# Installation
# ---------------------------------------------------------------------- 

echo "-----------------------------------------------------------------"
echo "Installing libraries"
echo "-----------------------------------------------------------------"
echo

# Change to library directory
cd ${LAGRANTO}/lib

# Loop over all libraries - compile and make library
set libs_load = ""
foreach lib ( $libs )

\rm -f ${lib}.a
\rm -f ${lib}.o
echo ${FORTRAN} -c -O ${lib}.f
${FORTRAN} -c -O ${fdebug} ${NETCDF_INC} ${lib}.f
ar r ${lib}.a ${lib}.o
\rm -f ${lib}.l ${lib}.o
ranlib ${lib}.a
set libs_load = "${libs_load} ${LAGRANTO}/lib/${lib}.a"
if ( ! -f ${lib}.a ) then
  echo "Problem in compiling ${lib} ... Stop"
  exit 1
endif

end

echo "-----------------------------------------------------------------"
echo "Installing core program"
echo "-----------------------------------------------------------------"
echo

# Change to program directory
cd ${LAGRANTO}/prog

\rm -f caltra.o 
\rm -f caltra

echo "${FORTRAN} -c -O ${NETCDF_INC} caltra.f"
${FORTRAN} -c -O ${fdebug} ${NETCDF_INC} caltra.f
echo "${FORTRAN} -o caltra caltra.o ${libs_load} ${NETCDF_INC} ${NETCDF_LIB}"
${FORTRAN} -o caltra caltra.o ${fdebug} ${libs_load} ${NETCDF_INC} ${NETCDF_LIB}

if ( ! -f caltra ) then
  echo "Problem in compiling caltra ... Stop"
  exit 1
endif



