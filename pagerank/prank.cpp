#include <iostream>
#include <stdio.h>

using namespace std;

const int N = 5;
const int iterations = 1000;
const float d = 0.85;
const float M[N][N] = {
    {0,     0,      0,      0,  1},
    {0.5,   0,      0,      0,  0},
    {0.5,   0,      0,      0,  0},
    {0,     1,      0.5,    0,  0},
    {0,     0,      0.5,    1,  0}
};

const float _M[N * N] = { 
    0,      0,  0,      0,  1,  
    0.5,    0,  0,      0,  0, 
    0.5,    0,  0,      0,  0,
    0,      1,  0.5,    0,  0,
    0,      0,  0.5,    1,  0 };
float _M_hat[N * N];

float M_hat[N][N];
float v[N];
float _v[N];

void matMul(float a[][N], float b[]){
    float temp[N] = { 0, };
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            temp[i] += a[i][j] * b[j];
        }
    }

    for(int i = 0; i < N; i++){
        v[i] = temp[i];
    }
}

void mat_Mul(float a[], float b[]) {
    float temp[N] = { 0, };

    for(int i = 0; i < N; i++) {
    	for(int j = 0; j < N; j++) {
    		temp[i] += a[i * N + j] * b[j];
    	}
    }

    for(int i = 0; i < N; i++)
    	_v[i] = temp[i];
}

int main()
{
    
    srand(time(0));
    int sum = 0;
    int _sum = 0;
    for(int i = 0; i < N; i++){
        v[i] = rand() % 1000;
        _v[i] = rand() % 1000;
        sum += v[i]; 
        _sum += _v[i];
    }

    for(int i = 0; i < N; i++){
       v[i] = v[i] / sum;
       _v[i] = _v[i] / _sum;
    }
    
    for(int i = 0; i < N; i++) {
        cout << v[i] << "\n";
        
    }

    float dd = (1 - d) / N;
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            M_hat[i][j] = d * M[i][j] + dd;
        }
    }

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            _M_hat[i * N + j] = d * _M[i * N + j] + dd;
        }
    }

    cout << "M_hat\n";
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            cout << M_hat[i][j] << ' ';
        }
        cout << "\n";
    }

    cout << "_M_hat\n";
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            cout << _M_hat[i * N + j] << ' ';
        }
        cout << "\n";
    }


    for(int i = 0; i < iterations; i++) {
        matMul(M_hat, v);
    }

    for(int i = 0; i < iterations; i++) {
        mat_Mul(_M_hat, _v);
    }
   
    float dsum = 0;
    cout << "result\n";
    for(int i = 0; i < N; i++){
       printf("%.6lf\n", v[i]);
       dsum += v[i];
    }
    cout << "sum: " << dsum << "\n";

    float _dsum = 0;
    cout << "2nd result\n";
    for(int i = 0; i < N; i++){
       printf("%.6lf\n", _v[i]);
       _dsum += _v[i];
    }
    cout << "sum: " << _dsum << "\n";

    return 0;
}