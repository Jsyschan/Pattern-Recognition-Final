
#include <iostream>
#include <map>
#include <vector>
#include <cmath>
using namespace std;

#include "Matrix.h"
#include "Pr.h"

int bestK(const Matrix &train,
          const Matrix &test,
          const int &numFeatures,
          const int &numClasses,
          const int &minkowskiDegree)
{
  int k = 1;
  int maxK = (int) floor( sqrt( train.getRow() ) );
  int bestK = 1;

  double start = 0;
  double end = 0;
  double duration = 0;

  bool partialDistance = true;

  Matrix knnData;

  Matrix predicted;
  Matrix correct;

  Matrix confuse;
  double sens = 0;
  double spec = 0;
  double prec = 0;
  double accu = 0;
  double error = 0;
  Matrix evaluations(1, 5);

  // Matrix degrees(3, 1);
  // degrees(0, 0) = 1;
  // degrees(1, 0) = 2;
  // degrees(2, 0) = maxK;

  Matrix degrees(1, 1);
  degrees(0, 0) = minkowskiDegree;
  double degree = 0;

  // error -> k
  map < double, int > errors;

  for ( int l = 0; l < degrees.getRow(); l++ )
  {
    degree = degrees(l, 0);

    for ( k = 1; k <= maxK; k++ )
    {
      cout << "\tk: " << k << endl << endl;
      cout << "\tminkowski degree: " << degree << endl;
      start = getTime();
      knnData = knn(train, test, k, numFeatures, numClasses, !partialDistance, degree);
      end = getTime();
      duration = end - start;
      cout << "\tduration: " << duration << endl;

      start = getTime();
      knnData = knn(train, test, k, numFeatures, numClasses, partialDistance, degree);
      end = getTime();
      duration = end - start;
      cout << "\tduration w/ partial distance: " << duration << endl << endl;

      predicted = subMatrix(knnData, 0, numFeatures, knnData.getRow() - 1, numFeatures);
      correct = subMatrix(test, 0, numFeatures, test.getRow() - 1, numFeatures);
      confuse = confusion(correct, predicted);

      perfEval(confuse, sens, spec, prec, accu);
      error = 1 - accu;

      cout << "\tsensitivity: " << sens << endl;
      cout << "\tspecificity: " << spec << endl;
      cout << "\tprecision: " << prec << endl;
      cout << "\taccuracy: " << accu << endl;
      cout << "\terror: " << error << endl << endl;

      evaluations(0, 0) = sens;
      evaluations(0, 1) = spec;
      evaluations(0, 2) = prec;
      evaluations(0, 3) = accu;
      evaluations(0, 4) = error;

      cout << "LaTeX Confusion Matrix" << endl;
      cout << toLatex(confuse) << endl;

      cout << "LaTeX Performance Evaluations" << endl;
      cout << toLatex(evaluations) << endl;

      errors[ error ] = k;
    }

    bestK = errors.begin()->second;
    cout << endl;
    cout << "best k: " << bestK << endl;
    cout << "error: " << errors.begin()->first << endl << endl;
  }

  return bestK;
}

Matrix knn(const Matrix &train,
           const Matrix &test,
           const int &k,
           const int &numFeatures,
           const int &numClasses,
           const bool &partialDistance,
           const int &minkowskiDegree)
{
  Matrix knnTest = test;

  Matrix testSample;
  int label = 0; 
  for ( int i = 0; i < knnTest.getRow(); i++ )
  {
    testSample = subMatrix(knnTest, i, 0, i, numFeatures - 1);

    label = knnPerSample(train, testSample, k, numFeatures, numClasses, partialDistance);
    knnTest(i, numFeatures) = label;
  }  

  return knnTest;
}

int knnPerSample(const Matrix &train,
                 const Matrix &testSample,
                 const int &k,
                 const int &numFeatures,
                 const int &numClasses,
                 const bool &partialDistance,
                 const int &minkowskiDegree)
{
  Matrix trainSample;

  // distance -> label
  multimap <double, double> sortedDistances;
  multimap <double, double>::iterator it;
  pair <double, double> entry;

  double label = 0;
  double distance = 0;

  double sum = 0;
  double maxDistance = -1;
  bool earlyExit = false;

  // int count = 0;
  vector <int> votes;
  votes.resize(numClasses, 0);

  int maxLabel = 0;
  int maxCount = -1;

  for ( int i = 0; i < train.getRow(); i++ )
  {
    trainSample = subMatrix(train, i, 0, i, numFeatures - 1);

    // Partial Distance Calculation
    // Notice: If partial distance is FALSE, produces same result as euc(trainSample, testSample)
    sum = 0;
    maxDistance = !( sortedDistances.empty() ) ? sortedDistances.rbegin()->first : -1;
    for ( int j = 0; j < trainSample.getCol(); j++ )
    {
      // sum += (double) pow( ( trainSample(0, j) - testSample(0, j) ), 2 );
      // distance = sqrt(sum);

      sum += (double) pow( ( trainSample(0, j) - testSample(0, j) ), minkowskiDegree );
      distance = (double) pow( sum, ( 1 / ( (double) minkowskiDegree ) ) );

      earlyExit = ( sortedDistances.size() == k ) && ( distance > maxDistance );
      if ( earlyExit && partialDistance )
        break;
    }

    // distance = euc(trainSample, testSample);
    label = train(i, numFeatures);
    entry = make_pair(distance, label);
    
    // Only keep the k smallest distances
    sortedDistances.insert(entry);
    if ( sortedDistances.size() > k )
      sortedDistances.erase( --sortedDistances.end() );
  }

  // Count votes
  // NOTICE: Assumes labels go from 0,...,numClasses-1
  for ( it = sortedDistances.begin(); it != sortedDistances.end(); ++it)
    votes[ it->second ]++;

  // Get the label with the highest number of votess
  for ( int i = 0; i < votes.size(); i++ )
  {
    if ( votes[i] > maxCount )
    {
      maxCount = votes[i];
      maxLabel = i;
    }
  }

  return maxLabel;
}
