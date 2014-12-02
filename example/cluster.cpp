
/**
 * @brief Clustering
 * @details Performs k-Means, winner-takes-all, and kohonen maps on passed in csv
 * 
 * @param argc [description]
 * @param argv [description]
 * 
 * @return [description]
 */

#include <cstdlib>
#include <iostream>
#include <string>
using namespace std;

#include "Matrix.h"
#include "Pr.h"

#define USAGE "./cluster dataFile numFeatures numClusters [learningRate [windowSize]]"
#define DEFAULT_LEARNING_RATE 0.01
#define DEFAULT_WINDOW_SIZE   0.01

int main(int argc, char **argv)
{
  if ( argc != 4 && argc != 5 && argc != 6 )
  {
    cerr << USAGE << endl;
    exit(EXIT_FAILURE);
  }
  
  string dataFile     = argv[1];
  int numFeatures     = atoi(argv[2]);
  int numClusters     = atoi(argv[3]);
  double learningRate = argc >= 5 ? atof(argv[4]) : DEFAULT_LEARNING_RATE;
  double windowSize   = argc == 6 ? atof(argv[5]) : DEFAULT_WINDOW_SIZE;

  double start = 0.0;
  double end = 0.0;
  double duration = 0.0;

  double mse = 0.0;

  Matrix data = readData(dataFile.c_str(), numFeatures);
  Matrix wineData = subMatrix(data, 0, 0, data.getRow() - 1, data.getCol() - 2);
  

  // cout << "Beginning kMeans" << endl;
  start = getTime();

  Matrix clusters = kMeans(wineData, 
                           numClusters);
  
  end = getTime();
  duration = end - start;
  // cout << "Ending kMeans" << endl;
  cout << "kMeans time: " << duration << endl;
    
  Matrix kMeansData = applyClusters(wineData, clusters);

  mse = meanSquaredError(wineData, kMeansData);
  cout << "Mean squared error (average over all samples): " << mse << endl;


  // cout << "Beginning winner-takes-all" << endl;
  start = getTime();

  Matrix wtaClusters = kMeans(wineData, 
                              numClusters, 
                              learningRate);

  end = getTime();
  duration = end - start;
  // cout << "Ending winner-takes-all" << endl;
  cout << "winner-takes-all time: " << duration << endl;

  Matrix wtaData = applyClusters(wineData, clusters);

  mse = meanSquaredError(wineData, wtaData);
  cout << "Mean squared error (average over all samples): " << mse << endl;


  // cout << "Beginning kohonen maps" << endl;
  start = getTime();

  Matrix kohonenClusters = kMeans(wineData, 
                                  numClusters, 
                                  learningRate,
                                  windowSize);

  end = getTime();
  duration = end - start;
  // cout << "Ending kohonen maps" << endl;
  cout << "kohonen maps time: " << duration << endl;

  Matrix kohData = applyClusters(wineData, clusters);

  mse = meanSquaredError(wineData, kohData);
  cout << "Mean squared error (average over all samples): " << mse << endl;

  return EXIT_SUCCESS;
}
