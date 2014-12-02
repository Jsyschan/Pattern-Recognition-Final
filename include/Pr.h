/*
 * pr.h - header file of the pattern recognition library
 *
 * Author: Hairong Qi, ECE, University of Tennessee
 *
 * Date: 01/25/04
 *
 * Please send all your comments to hqi@utk.edu 
 * 
 * Modified:
 *   - 09/24/13: add "const" to the filename parameters to remove warning 
 *               msg in new compilers (Steven Clukey)
 *   - 04/26/05: reorganized for the Spring 2005 classs
 */

#ifndef _PR_H_
#define _PR_H_

#include "Matrix.h"

/////////////////////////  
// file I/O
Matrix readData(const char *,            // the file name
                int);                    // the number of columns of the matrix
Matrix readData(const char *,            // the file name
                int,                     // the number of columns
                int);                    // the number of rows (or samples)
Matrix readData(const char *);           // read data file to a matrix with 1 row
void writeData(Matrix &, const char *);  // write data to a file
Matrix readImage(const char *,           // read the image from a file
                 int *,                  // the number of rows (or samples)
                 int *);                 // the number of columns
void writeImage(const char *,            // write the image to a file
                Matrix &,                // the matrix to write
                int,                     // the number of rows
                int);                    // the number of columns


////////////////////////
// distance calculation
double euc(const Matrix &,         // Euclidean distance between two vectors
	   const Matrix &);
double mah(const Matrix &,         // the Mahalanobis distance, input col vec
	   const Matrix &C,        // the covariance matrix
	   const Matrix &mu);      // the mean (a col vector)

// JCW
double minkowski(const Matrix &trainSample,
                 const Matrix &testSample,
                 int &degree);

// Normalize training data (Qi)
void normalize(Matrix &tr, 
          Matrix &te, 
          const int numFeatures, 
          const int flag); // 1 for supervised, 0 for unsupervised

// Normalized data (JCW)
void normalizeParams(const Matrix &data,
                       Matrix &mu,
                       Matrix &sigma,
                       const int &numFeatures);

Matrix normalize(const Matrix &data,
                 const Matrix &mu,
                 const Matrix &sigma,
                 const int &numFeatures);

////////////////////////
// prior probablities (JCW)

// return a column vector (n x 1) of random prior probablities
Matrix randomPriors(const int &numClasses);
// Matrix random(const int &count);

////////////////////////
// classifiers

// maximum a-posteriori probability (MPP)
int mpp(const Matrix &train,        // the training set of dimension mx(n+1)
                                    // where the last col is the class label
                                    // that starts at 0
        const Matrix &test,         // one test sample (a col vec), nx1
        const int,                  // number of different classes
    	const int,                  // caseI,II,III of the discriminant func
    	const Matrix &Pw);          // the prior prob, a col vec


// k-nearest neighbor (kNN) (JCW)
#define PART_DIST false
#define DEGREE 2

int bestK(const Matrix &train,
          const Matrix &test,
          const int &numFeatures,
          const int &numClasses,
          const int &minkowskiDegree);

Matrix knn(const Matrix &train,
           const Matrix &test,
           const int &k,
           const int &numFeatures,
           const int &numClasses,
           const bool &partialDistance = PART_DIST,
           const int &minkowskiDegree = DEGREE);

int knnPerSample(const Matrix &train,
                 const Matrix &testSample,
                 const int &k,
                 const int &numFeatures,
                 const int &numClasses,
                 const bool &partialDistance = PART_DIST,
                 const int &minkowskiDegree = DEGREE);

////////////////////////
// dimensionality reduction (JCW)

// principal component analysis (PCA) (Qi)
int pca(Matrix &tr, 
        Matrix &te, 
        const int numFeatures, 
        const float err, 
        const int flag);

// principal component analysis (PCA) (JCW)
Matrix pca(const Matrix &normalizedData, // m x (n+1) i.e. w/ classification
           const double &maxError,
           const int &numFeatures);

// fisher's linear discriminant (FLD)
Matrix fld(const Matrix &normalizedData, // m x (n+1) i.e. w/ classification
           const int &numFeatures);

////////////////////////
// Performance Evaluation (JCW)

// Confusion Matrix
Matrix confusion(const Matrix &actualLabels,
                 const Matrix &predictedLabels);

void perfEval(const Matrix &confusion,
              double &sensitivity,
              double &specificity,
              double &precision,
              double &accuracy);

double classificationError(const Matrix &actualLabels,
                           const Matrix &predictedLabels);

double meanSquaredError(const Matrix &real, const Matrix &pred);

////////////////////////
// Computation Time (JCW)
double getTime();

////////////////////////
// Clustering
Matrix kMeans(const Matrix &data, 
              const int &numClusters, 
              const double &learningRate = 0.0,
              const double &windowSize = 0.0);

Matrix applyClusters(const Matrix &data, 
                     const Matrix &clusters);

#endif

