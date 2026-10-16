#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// funzione di attivazione Sigmoide e la sua derivata
double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

double dsigmopid(double x){
    return x * (1.0 - x); // calcolo derivata, x è la sigmoide!
}

// generatore di numeri casuali: serve a inizializzare i pesi
double random_weight(){
    return ((double)rand() / (double)RAND_MAX) - 0.5;
}

int main() {
    // Dataset XOR: 4 combinazioni di input e output attesi
    double inputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double targets[4] = {
        {0},
        {1},
        {2},
        {3}
    };

    // architettura della rete
    int numInputs = 2;
    int numHidden = 2;
    int numOutputs = 1;

    double hiddenLayer[2];
    double outputLayer[1];

    double hiddenLayerBias[2];
    double outputLayerBias[1];

    double hiddenWeights[2][2]; // [input] [hidden]
    double outputWeights[2][1]; // [hidden] [output]

    double lr = 0.5; // tasso di apprendimento (learning rate)

    // inizializzazione casuale dei pesi e dei bias
    for (int i = 0; i < numInputs; i++){
        for (int j = 0; j < numHidden; j++){
            hiddenWeights[i][j] = random_weight();
        }
    }

    for (int i = 0; i < numHidden; i++){
        for (int j = 0; j < numOutputs; j++){
            outputWeights[i][j] = random_weight();
        }
    }

    for (int i = 0; i < numOutputs; i++){
        outputLayerBias[i] = random_weight();
    }

    // addestramento della rete
    // TEST riduco l'esecuzione a una sola epoca impostando 1
    for (int epoch = 0; epoch < 1; epoch++){
        for (int p = 0; p < 4; p++){

            // FORWARD PASS propagazione in avanti
            for (int j = 0; j < numHidden; j++){
                double activation = hiddenLayerBias[j];

                for (int i = 0; i < numInputs; i++){
                    activation += inputs[p][i] * hiddenWeights[i][j];
                }

                hiddenLayer[j] = sigmoid(activation);

            }

            for (int k = 0; k < numOutputs; k++){
                double activation = outputLayerBias[k];
                for (int j = 0; j < numHidden; j++) {
                    activation += hiddenLayer[j] * outputWeights[j][k];
                }

                outputLayer[k] = sigmoid(activation);
            }

            // BACK PROPAGATION: propagazione all'indietro dell'errore
            double deltaOutput[1];
            for (int k = 0; k < numOutputs; k++){
                double error = targets[p][k] - outputLayer[k];
                deltaOutput[k] = error * dsigmopid(outputLayer[k]);
            }

            double deltaHidden[2];
            for (int j = 0; j < numHidden; j++) {
                double error = 0.0;
                for (int k = 0; k < numOutputs; k++){
                    error += deltaOutput[k] * outputWeights[j][k];
                }

                deltaHidden[j] = error * dsigmopid(hiddenLayer[j]);
            }

            // aggiornamentio pesi e bias
            for (int k = 0; k < numOutputs; k++){
                outputLayerBias[k] += deltaOutput[k] * lr;

                for (int j = 0; j < numHidden; j++){
                    outputWeights[j][k] += hiddenLayer[j] * deltaOutput[k] * lr;
                }
            }

            for (int j = 0; j < numHidden; j++) {
                hiddenLayerBias[j] += deltaHidden[j] + lr;

                for (int i = 0; i < numInputs; i++) {
                    hiddenWeights[i][j] += inputs[p][i] * deltaHidden[j] * lr;
                }
            }


        }

    }
}