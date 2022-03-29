#include <stdio.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <iostream>

using namespace std;

void print(const char name[], vector<float> mat, int row, int col) {
    printf("\n%s: \n", name);

    for(int r = 0; r < row; r++){
        for(int c = 0; c < col; c++) {
            printf("%.6f ", mat[r * col + c]);
        }
        printf("\n");
    }
}

void matMul(vector<float> a, vector<float> &b, int row, int col) {
    vector<float> temp(col, 0);

    for(int i = 0; i < row; i++) {
    	for(int j = 0; j < col; j++) {
    		temp[i] += a[i * col + j] * b[j];

    	}
    }

    for(int i = 0; i < col; i++)
    	b[i] = temp[i];
}

int main(int argc, char **argv) {
    int rows, columns, iters;
    const float diff = 0.01;
    const float d = 0.85;
    
    freopen(argv[1], "r", stdin);

    scanf("%d %d %d", &rows, &columns, &iters);

    vector<float> M(rows * columns);
    vector<float> V(columns);
    vector<float> gold(columns);


    for(int r = 0; r < rows; r++)
        for(int c = 0; c < columns; c++)
            scanf("%f", &M[r* columns + c]);

    for(int c = 0; c < columns; c++)
        scanf("%f", &V[c]);

    for(int c = 0; c < columns; c++)
        scanf("%f", &gold[c]);


    //print("M", M, rows, columns);
    //print("V", V, rows,  1);

    for(int i = 0; i < rows * columns; i++) {
		M[i] = d * M[i] + (1-d) / columns;
	}
        
 
    //print("M hat", M, rows, columns);

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    for(int i = 0; i < iters; i++) {
        matMul(M, V, rows, columns);
    }
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
	std::chrono::nanoseconds nano = end - start;
   
    //print("result", V, rows, 1);

    printf("\n----------result---------\n");
    printf("----gold----|---result---\n");
    bool ok = true;
    for(int c = 0; c < columns;c++) {
        printf("%-12.10f|%-12.10f ", gold[c], V[c]);

        printf("%c\n", (abs(gold[c] -  V[c]) >= diff) ? 'X' : 'O');
    }

    cout << "N : " << rows << "\n";
    cout << "Iterations : " << iters << '\n';
    cout << "Host execution time : " << nano.count() << " \n";
    
    printf("%s\n", ok ? "ok" : "wrong");
    
    return 0;
}