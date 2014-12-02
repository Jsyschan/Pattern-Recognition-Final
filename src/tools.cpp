
#include <cstdlib>
#include <cmath>
#include <string>
#include <sstream>
#ifdef _WIN32
	//#include <WinBase.h>
	#include <time.h>
#else
	#include <sys/time.h>
#endif


#include "Matrix.h"
#include "Pr.h"

void normalizeParams(const Matrix &data,
											 Matrix &mu,
											 Matrix &sigma,
											 const int &numFeatures)
{
	mu = mean(data, numFeatures);
	sigma.createMatrix(numFeatures, 1);
	Matrix covariance = cov(data, numFeatures);
	for ( int i = 0; i < sigma.getRow(); i++ )
		sigma(i, 0) = sqrt( covariance(i, i) );
}

Matrix normalize(const Matrix &data,
								 const Matrix &mu,
								 const Matrix &sigma,
								 const int &numFeatures)
{
	Matrix normData = data;

	for ( int i = 0; i < normData.getRow(); i++ )
	{
		for ( int j = 0; j < numFeatures; j++ )
			normData(i, j) = ( normData(i, j) - mu(j, 0) ) / sigma(j, 0);
	}

	return normData;
}

Matrix randomPriors(const int &numClasses)
{
	static bool first = 1;
	if ( first )
	{
		srand(time(NULL));
		first = false;
	}
	
	Matrix priorProbs(numClasses, 1);
	int sum = 0;

	for ( int i = 0; i < numClasses; i++ )
  {
    priorProbs(i, 0) = rand() % 1000 + 1;
    sum += priorProbs(i, 0);
  }

  return priorProbs / (double) sum;
}

// Confusion matrix
// Form:
//   TP FN
//   FP TN
Matrix confusion(const Matrix &actualLabels,
							  const Matrix &predictedLabels)
{
	if ( actualLabels.getRow() != predictedLabels.getRow() )
	{
		cerr << "Same # of samples needed for perfEval" << endl;
		exit(EXIT_FAILURE);
	}

	Matrix confusion(2, 2);

	bool TP = false; // True positive
	bool TN = false; // True negative
	bool FP = false; // False positive
	bool FN = false; // False negative

	for ( int i = 0; i < actualLabels.getRow(); i++ )
	{
		TP = actualLabels(i, 0) == 1 && predictedLabels(i, 0) == 1;
		TN = actualLabels(i, 0) == 0 && predictedLabels(i, 0) == 0;
		FP = actualLabels(i, 0) == 0 && predictedLabels(i, 0) == 1;
		FN = actualLabels(i, 0) == 1 && predictedLabels(i, 0) == 0;

		if ( TP )
			confusion(0, 0)++;	
		else if ( FN )
			confusion(0, 1)++;
		else if ( FP )
			confusion(1, 0)++;	
		else if ( TN )
			confusion(1, 1)++;
	}

	return confusion;
}

void perfEval(const Matrix &confusion,
							double &sensitivity,
							double &specificity,
							double &precision,
							double &accuracy)
{
	if (confusion.getRow() != 2 || confusion.getCol() != 2)
	{
		cerr << "Confusion matrix must be a 2 x 2 in perfEval" << endl;
		exit(EXIT_FAILURE);
	}

	double TP = confusion(0, 0);
	double FN = confusion(0, 1);
	double FP = confusion(1, 0);
	double TN = confusion(1, 1);

	sensitivity = TP / ( TP + FN );
	specificity = TN / ( TN + FP );
	precision = TP / ( TP + FP );
	accuracy = ( TP + TN ) / ( TP + TN + FP + FN ); 
}

Matrix appendColumn(const Matrix &data, 
										const Matrix &column)
{
	if ( data.getRow() != column.getRow() )
	{
		cerr << "# rows must be the same to append" << endl;
		exit(EXIT_FAILURE);
	}

	Matrix newData(data.getRow(), data.getCol() + 1);
	int i = 0;
	int j = 0;

	for ( i = 0; i < data.getRow(); i++ )
	{
		for ( j = 0; j < data.getCol(); j++ )
		{
			newData(i, j) = data(i, j);
		}
		newData(i, j) = column(i, 0);
	}

	return newData;
}

double classificationError(const Matrix &actualLabels,
						 							 const Matrix &predictedLabels)
{
	if ( actualLabels.getRow() != predictedLabels.getRow() )
	{
		cerr << "Same # of samples needed for classificationError()" << endl;
		exit(EXIT_FAILURE);
	}

	double incorrectCount = 0;
	for (int i = 0; i < actualLabels.getRow(); i++ )
	{
		if ( actualLabels(i, 0) != predictedLabels(i, 0) )
			incorrectCount++;
	}

	return incorrectCount / (double) actualLabels.getRow();
}

string toLatex(const Matrix &data)
{
	string s = "";
	bool notLastCol = false;

	ostringstream buffer;
	string value = "";

	for ( int i = 0; i < data.getRow(); i++ )
	{
		for (int j = 0; j < data.getCol(); j++ )
		{
			buffer.str("");
			buffer.clear();
			buffer << data(i, j);
			value = buffer.str();

			notLastCol = j < data.getCol() - 1;
			s += ( notLastCol ? ( value + " & " ) : value );
		}
		s += "\n";
	}

	return s;
}

double getTime()
{
	#ifndef _WIN32
		struct timeval time;
		if (gettimeofday(&time, NULL)) // http://linux.die.net/man/3/gettimeofday
				exit(EXIT_FAILURE);
		return (double) time.tv_sec + (double) time.tv_usec * .000001;
	#else
		return (double)clock() / CLOCKS_PER_SEC;
	#endif
}

double vectorNorm(const Matrix &data)
{
	if ( data.getCol() != 1 )
	{
		cerr << "Can only get the vector norm of 1 column matrices" << endl;
		exit(EXIT_FAILURE);
	}

	double sum = 0;
	int i = 0;
	for ( i = 0; i < data.getRow(); i++ )
		sum += pow( data(i, 0), 2 );
	return sqrt( sum );
}

double meanSquaredError(const Matrix &real, const Matrix &pred)
{
	if ( ( real.getRow() != pred.getRow() ) && ( real.getCol() != pred.getCol() ) )
	{
		cerr << "Cannot calculate mean squared error. Matrices must have the same dimensions." << endl;
		exit(EXIT_FAILURE);
	}

	Matrix errors( real.getRow(), 1 );
	int i = 0;
	int j = 0;
	double diffSum = 0.0;
	double errorSum = 0.0;

	for ( i = 0; i < real.getRow(); i++ )
	{
		for ( j = 0; j < real.getCol(); j++ )
		{
			diffSum += pow( pred(i, j) - real(i, j), 2);
			// cout << "diff: " << diffSum << endl;
		}

		errors(i, 0) = ( (double) 1.0 / real.getCol() ) * diffSum;
		// cout << "Error " << i << ": " << errors(i, 0)  << endl;
		errorSum += errors(i, 0);
		diffSum = 0.0;
	}

	return errorSum / real.getRow();
}

