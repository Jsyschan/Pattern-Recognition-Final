/**********************************************************
 * preprocessing.cpp  
 *
 *   - normalize: normalize training and test set
 *   - pca: principal component analysis
 * 
 * Author: Hairong Qi (C) hqi@utk.edu
 *
 **********************************************************/
#include "Matrix.h"
#include "Pr.h"
#include <iostream>
#include <cstdlib>
#include <cmath>

using namespace std;


/**
 * Matrix normalization.
 * @param tr The training set.
 * @param te The test set. 
 * @param numFeatures The number of features.
 * @param flag If flag is on, it's supervised learning; otherwise, it's
 *             unsupervised learning and the second argument can be empty.
 */
void normalize(Matrix &tr, Matrix &te, const int numFeatures, const int flag)
{
  Matrix mu, Sigma, sigma;

  // get the statistics from the training set
  mu = mean(tr, numFeatures);

  Sigma = cov(tr, numFeatures);
  sigma.createMatrix(numFeatures,1);
  for (int j=0; j<numFeatures; j++)
    sigma(j,0) = sqrt(Sigma(j,j));

  // normalize the training set
  for (int i=0; i<tr.getRow(); i++) {
    for (int j=0; j<numFeatures; j++)
      tr(i,j) = (tr(i,j)-mu(j,0)) / sigma(j,0);
  }

  // normalize the test set
  if (flag) {
    for (int i=0; i<te.getRow(); i++) {
      for (int j=0; j<numFeatures; j++)
      	te(i,j) = (te(i,j)-mu(j,0)) / sigma(j,0);
    }
  }
}


/**
 * Principal component analysis.
 * @param tr The training set.
 * @param te The test set. 
 * @param numFeatures The number of features.
 * @param err The error rate needs to be satisfied.
 * @param flag If flag is on, it's supervised learning; otherwise, it's
 *             unsupervised learning and the second argument can be empty.
 * @return The number of features after PCA based on "err"
 */
int pca(Matrix &tr, Matrix &te, const int numFeatures, const float err, const int flag)
{
  Matrix Sigma, sampleRow;
  Matrix eigenvalue(1,numFeatures),   // eigenvalue (a row vector) 
    eigenvector(numFeatures,numFeatures),       // eigenvector with each col an eigenvector
    pcaEig;             // eigenvectors selected based on "err"
  int p, numReducFeatures;
  float psum, sum;

  Sigma = cov(tr, numFeatures);
  jacobi(Sigma, eigenvalue, eigenvector);
  eigsrt(eigenvalue, eigenvector);   // sort the eigenvalue in the ascending order

  // determine the number of principal components to keep based on "err" given
  sum = 0.0;
  for (int i=0; i<numFeatures; i++)
    sum += eigenvalue(0,i);

  p = 0;
  psum = 0.0;
  while (psum/sum < err && p < numFeatures) {
    psum += eigenvalue(0,p);
    p++;
  }
  numReducFeatures = numFeatures - p;

  pcaEig = subMatrix(eigenvector,0,p,numFeatures-1,numFeatures-1);
  
  // perform the transformation 
  for (int i=0; i<tr.getRow(); i++) {          // for training set
    sampleRow = subMatrix(tr,i,0,i,numFeatures-1);
    sampleRow = sampleRow ->* pcaEig;
    for (int j=0; j<numReducFeatures; j++)
      tr(i,j) = sampleRow(0,j);
  }
    
  if (flag) {
    for (int i=0; i<te.getRow(); i++) {          // for test set
      sampleRow = subMatrix(te,i,0,i,numFeatures-1);
      sampleRow = sampleRow ->* pcaEig;
      for (int j=0; j<numReducFeatures; j++)
      	te(i,j) = sampleRow(0,j);
    }
  }

  return numReducFeatures;
}
