
#include <cfloat>
#include <string>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include "Matrix.h"
#include "Pr.h"

Matrix kMeans(const Matrix &data, 
              const int &numClusters, 
              const double &learningRate,  // Winner-takes-all
              const double &windowSize)    // Kohonen Maps
{
  Matrix dataCopy;
  Matrix classColumn;
  Matrix clusters;
  Matrix sample( 1, data.getCol() );
  Matrix cluster( 1, data.getCol() ) ;
  int i = 0;
  int j = 0;
  int m = 0;
  int step = 0;
  double minDistance = DBL_MAX;
  double distance = 0;
  int closestCluster = 0;
  int currentCluster = 0;
  int numChanged = 1;
  Matrix samples;
  Matrix sampleMean;
  int iteration = 1;
  string filePrefix = "cluster_";
  string clusterFile = filePrefix;
  ostringstream outStream;
  // int seed = time(NULL);
  // int seed = 1416450784;
  // int randIndex = 0;
  double numerator = 0.0;
  double denominator = 0.0;
  int dimen = (int) sqrt( numClusters );
  Matrix coord(2, 1);
  Matrix winner(2, 1);
  Matrix difference(2, 1);
  double result = 0.0;
  int iterationLimit = 1000;

  dataCopy = data;
  classColumn.createMatrix( data.getRow(), data.getCol() );
  dataCopy = appendColumn( dataCopy, classColumn );

  // Pick initial clusters (every n / numClusters samples)
  step = dataCopy.getRow() / numClusters;
  clusters.createMatrix( numClusters, dataCopy.getCol() - 1 );
  for ( i = 0; i < numClusters; i++ )
  {
    for ( j = 0; j < clusters.getCol(); j++ )
      clusters(i, j) = dataCopy(i * step, j);
  }

  // Pick initial clusters (first numClusters samples)
  // for ( i = 0; i < numClusters; i++ )
  // {
  //   for ( j = 0; j < clusters.getCol(); j++ )
  //     clusters(i, j) = dataCopy(i, j);
  // }

  // Pick initial clusters (random numClusters samples)
  // cout << "Seed: " << seed << endl;
  // srand( seed );
  // for ( i = 0; i < numClusters; i++ )
  // {
  //   randIndex = rand() % dataCopy.getRow();
  //   for ( j = 0; j < clusters.getCol(); j++ )
  //     clusters(i, j) = dataCopy(randIndex, j);
  // }


  while ( numChanged > 0 && iteration < iterationLimit )
  {
    numChanged = 0;

    // Assign samples to closest clusters
    for ( i = 0; i < dataCopy.getRow(); i++ )
    {
      for ( j = 0; j < sample.getCol(); j++ )
        sample(0, j) = dataCopy(i, j);
      
      minDistance = DBL_MAX;
      distance = minDistance;
      currentCluster = dataCopy( i, dataCopy.getCol() - 1 );
      closestCluster = currentCluster;

      for ( j = 0; j < clusters.getRow(); j++ )
      {
        for ( m = 0; m < cluster.getCol(); m++ )
          cluster(0, m) = clusters(j, m);

        distance = euc( sample, cluster );
        if ( distance < minDistance )
        {
          closestCluster = j;
          minDistance = distance;
        }
      }
      
      if ( currentCluster != closestCluster )
        numChanged++;

      dataCopy( i, dataCopy.getCol() - 1 ) = closestCluster;

      // Winner-Take All addition
      for ( j = 0; j < sample.getCol() && learningRate > 0.0 && windowSize == 0.0; j++ )
          clusters(closestCluster, j) = clusters(closestCluster, j) + learningRate * ( sample(0, j) - clusters(closestCluster, j) );
    
      // Kohonen Maps (Self-Organizing Maps: SOM)
      winner(0, 0) = closestCluster / dimen;                    // Row
      winner(1, 0) = closestCluster - ( winner(0, 0) * dimen ); // Col
      denominator = 2 * pow( windowSize, 2 );
      for ( j = 0; j < numClusters && learningRate > 0.0 && windowSize > 0.0; j++ )
      {
        // 2-d topology coordinates
        coord(0, 0) = j / dimen;                   // Row
        coord(1, 0) = j - ( coord(0, 0) * dimen ); // Col
        for ( m = 0; m < coord.getRow(); m++ )
          difference(m, 0) = coord(m, 0) - winner(m, 0);
        numerator = pow( vectorNorm( difference ), 2);
        result = exp( -1.0 * ( numerator / denominator) );

        for ( m = 0; m < clusters.getCol(); m++ )
          clusters(j, m) = clusters(j, m) + ( learningRate * result * ( sample(0, m) - clusters(j, m) ) );
      }

    }

    // Get sample mean for each cluster
    for ( i = 0; i < clusters.getRow(); i++ )
    {
      samples = getType( dataCopy, i );
      if ( samples.getRow() == 0 )
        continue;
      sampleMean = mean( samples, samples.getCol() );
      sampleMean = transpose( sampleMean );

      for ( j = 0; j < clusters.getCol(); j++ )
        clusters(i, j) = sampleMean(0, j);
    }

    outStream << filePrefix << iteration;
    clusterFile = outStream.str();
    outStream.str("");
    // clusterFile = filePrefix + iterationChar;
    writeData( clusters, clusterFile.c_str() ); 

    cout << "Iteration: " << iteration << endl;
    // cout << "NumChanged: " << numChanged << endl;

    iteration++;
  }

  return clusters;
}
