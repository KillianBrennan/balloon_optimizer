#!/bin/csh

module load dyn_tools

setenv NETCDF_FORMAT CF
setenv LAGRANTO ./lagranto.era5

${LAGRANTO}/prog/caltra startf 70 trajectory.4 -i 360 -o 10 -ts 1 -ref  20240917_18 

exit 0
