Pattern-Recognition-Final
=========================

Directory Structure
- .        : contains premake, README, and LICENSEs
- data     : contains wine quality data
- example  : contains driver files (*.cpp)
  - obj  : contains driver object files (*.o)
- include  : contains library headers (*.h)
- lib      : contains library file (*.a)
- proposal : contains project proposal 
- report   : contains final report
- src      : contains library source files (*.cpp)
  - obj  : contains library object files (*.o)


Wine Quality Data
- http://archive.ics.uci.edu/ml/datasets/Wine+Quality


Premake 4
- a portable build system
- default config file searched for is premake4.lua
  - can specify a different config file on the command line
- https://github.com/annulen/premake
- http://industriousone.com/what-premake

The following generates GNU Makefiles
./premake gmake
