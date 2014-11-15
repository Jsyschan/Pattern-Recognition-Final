
/**
 * Dimensionality Reduction
 * - PCA: Principal Component Analysis
 * - FLD: Fisher's Linear Discriminant
 * 
 * 
 */

#include <cmath>
#include "Pr.h"

Matrix pca(const Matrix &normalizedData,
           const double &maxError,
           const int &numFeatures)
{
  int n = normalizedData.getRow();
  Matrix eigenvalue(1, numFeatures);
  Matrix eigenvector(numFeatures, numFeatures);

  Matrix temp = subMatrix(normalizedData, 0, 0, n - 1, numFeatures - 1);
  Matrix scatter = (double) (n - 1) * cov(temp, numFeatures);
	
  jacobi(scatter, eigenvalue, eigenvector);
  eigsrt(eigenvalue, eigenvector);

  double eigenSum = 0;

  for ( int i = 0; i < numFeatures; i++ )
    eigenSum += eigenvalue(0, i);

  int i, j;
  double error = 0;
  for ( i = 0; i < numFeatures && error < maxError; i++ )
  {
    double sum = 0;
    for ( j = 0; j <= i; j++ )
      sum += eigenvalue(0, j);
    error = sum / eigenSum;
  }

	return subMatrix(eigenvector, 0, j - 1, eigenvector.getRow() - 1, eigenvector.getCol() - 1);
}

Matrix fld(const Matrix &normalizedData, 
           const int &numFeatures)
{
  Matrix w;
  Matrix scatter(numFeatures, numFeatures);

  Matrix class_1 = getType(normalizedData, 0);
  Matrix mean_1 = mean(class_1, numFeatures);
  int numSamples_1 = class_1.getRow();
  scatter += (numSamples_1 - 1) * cov(class_1, numFeatures);

  Matrix class_2 = getType(normalizedData, 1);
  Matrix mean_2 = mean(class_2, numFeatures);
  int numSamples_2 = class_2.getRow(); 
  scatter += (numSamples_2 - 1) * cov(class_2, numFeatures);

  Matrix meanDiff = ( mean_1 - mean_2 );
  
  w = inverse(scatter) ->* ( meanDiff );
  
  double magnitude = 0;
  for (int i = 0; i < w.getRow(); i++)
    magnitude += pow( w(i, 0), 2);
  magnitude = sqrt(magnitude);

  // Unit vector
  return w / magnitude;
}
