#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

const int iterations = 1000;
const int N = 5;

int main()
{
    MatrixXd M = MatrixXd(N, N);
    VectorXd v = VectorXd(M.cols());
    float d = 0.85;

    M << 0, 0, 0, 0, 1,
    0.5, 0, 0, 0, 0,
    0.5, 0, 0, 0, 0,
    0, 1, 0.5, 0, 0,
    0, 0, 0.5, 1, 0;

    srand(time(0));
    int sum = 0;
    for(int i = 0; i < M.cols(); i++){
        v(i) = rand() % 1000;
        sum += v(i);
    }

    for(int i = 0; i < M.cols(); i++){
        v(i) = v(i) / sum;
    }

    MatrixXd temp = MatrixXd(5, 5);
    float dd = (1 - d) / M.cols();
    for(int i = 0; i < M.rows(); i++) {
        for(int j = 0; j < M.cols(); j++) {
            temp(i, j) = dd;
        }
    }

    MatrixXd M_hat = d * M + temp;
    for(int i = 0; i < iterations; i++) {
        v = M_hat * v;
    }
    cout << v << endl;
   
    return 0;
}
