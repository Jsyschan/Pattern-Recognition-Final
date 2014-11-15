
#include <cstdlib>
#include <cmath>
#include <iostream>
using namespace std;

#include "Pr.h"
#include "Matrix.h"

double minkowski(const Matrix &trainSample,
                 const Matrix &testSample,
                 int &degree)
{
  if ( trainSample.getCol() != testSample.getCol() )
  {
    cerr << "samples do not have the same # of features" << endl;
    exit(EXIT_FAILURE);
  }

  double sum = 0;
  double difference = 0;

  for ( int i = 0; i < trainSample.getCol(); i++ )
  {
    difference = trainSample(0, i) - testSample(0, i);
    sum += (double) pow( difference, degree );
  }

  return (double) pow( sum, (1 / (double) degree) );
}
