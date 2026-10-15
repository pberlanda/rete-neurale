#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

double sigmoide(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double sigmoide_parametrica(double z) {
    double A = 1.0;
    double K = 1.0;
    return A / (1.0 + exp(-K * z));
}

int main(){
    double x[] = {0.5, 1.0, 1.5, 2.0, 2.5, 3.0};
    int n = sizeof(x) / sizeof(x[0]);

    srand(time(NULL));
    double w = (double)rand() / RAND_MAX * 2 - 1;
    double b = (double)rand() / RAND_MAX * 2 - 1;

    for (int i = 0; i < n; i++) {
        double z = x[i] * w + b;
        double y1 = sigmoide(z);
        double y2 = sigmoide_parametrica(z);

        printf("%d\t%f\t%f\t%f\t%f\n", i, x[i], z, y1, y2);
    }

    return 0;


}
