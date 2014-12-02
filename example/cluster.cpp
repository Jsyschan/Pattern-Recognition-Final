
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
#define DEFAULT_LEARNING_RATE 0.1
#define DEFAULT_WINDOW_SIZE   0.1

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

  Matrix data = readData(dataFile.c_str(), numFeatures);
  
  cout << "Beginning kMeans" << endl;
  start = getTime();

  Matrix clusters        = kMeans(data, 
                                  numClusters);
  
  end = getTime();
  duration = end - start;
  cout << "Ending kMeans" << endl;
  cout << "kMeans time: " << duration << endl;
  
  Matrix kMeansData = applyClusters(data, clusters);
  // Get error


  cout << "Beginning winner-takes-all" << endl;
  start = getTime();

  Matrix wtaClusters     = kMeans(data, 
                                  numClusters, 
                                  learningRate);

  end = getTime();
  duration = end - start;
  cout << "Ending winner-takes-all" << endl;
  cout << "winner-takes-all time: " << duration;

  // Matrix wtaData = applyClusters(data, clusters);
  // Get error

  cout << "Beginning kohonen maps" << endl;
  start = getTime();

  Matrix kohonenClusters = kMeans(data, 
                                  numClusters, 
                                  learningRate,
                                  windowSize);

  end = getTime();
  duration = end - start;
  cout << "Ending kohonen maps" << endl;
  cout << "kohonen maps time: " << duration;

  // Matrix kohData = applyClusters(data, clusters);
  // Get error

  return EXIT_SUCCESS;
}
