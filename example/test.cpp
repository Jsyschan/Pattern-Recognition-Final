
#include <cstdlib>
#include <iostream>
using namespace std;

#include "Matrix.h"

int main(int argc, char **argv)
{
  cout << "Hello, world!" << endl;

  Matrix x(2, 2);
  x(0, 0) = 10;
  x(1, 1) = 10;

  cout << x;

  return EXIT_SUCCESS;
}
