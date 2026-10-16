#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Funzione di attivazione Sigmoide e la sua derivata
double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

double dsigmoid(double x){
    return x * (1.0 - x); // x è già il valore sigmoidato
}

// Generatore di numeri casuali tra -0.5 e 0.5
double random_weight(){
    return ((double)rand() / (double)RAND_MAX) - 0.5;
}

int main(void) {
    // Dataset XOR (Input e Target corretti)
    double inputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    double targets[4] = {0, 1, 1, 0}; // Corretto: per XOR è 0, 1, 1, 0

    // Architettura della rete
    int numInputs = 2;
    int numHidden = 2;
    int numOutputs = 1;

    double hiddenLayer[2];
    double outputLayer[1];

    double hiddenLayerBias[2];
    double outputLayerBias[1];

    double hiddenWeights[2][2]; // [input][hidden]
    double outputWeights[2][1]; // [hidden][output]

    double lr = 0.5; // Tasso di apprendimento (learning rate)

    // Inizializzazione casuale dei pesi e dei bias
    for (int i = 0; i < numInputs; i++){
        for (int j = 0; j < numHidden; j++){
            hiddenWeights[i][j] = random_weight();
        }
    }

    for (int j = 0; j < numHidden; j++){
        hiddenLayerBias[j] = random_weight(); // Aggiunta inizializzazione mancante
    }

    for (int i = 0; i < numHidden; i++){
        for (int j = 0; j < numOutputs; j++){
            outputWeights[i][j] = random_weight();
        }
    }

    for (int i = 0; i < numOutputs; i++){
        outputLayerBias[i] = random_weight();
    }

    // Addestramento della rete (impostato a 10000 epoche per farla convergere)
    for (int epoch = 0; epoch < 10000; epoch++){
        for (int p = 0; p < 4; p++){

            // FORWARD PASS
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

            // BACKPROPAGATION
            double deltaOutput[1];
            for (int k = 0; k < numOutputs; k++){
                double error = targets[p] - outputLayer[k]; // Accesso corretto a targets
                deltaOutput[k] = error * dsigmoid(outputLayer[k]);
            }

            double deltaHidden[2];
            for (int j = 0; j < numHidden; j++) {
                double error = 0.0;
                for (int k = 0; k < numOutputs; k++){
                    error += deltaOutput[k] * outputWeights[j][k];
                }
                deltaHidden[j] = error * dsigmoid(hiddenLayer[j]);
            }

            // AGGIORNAMENTO PESI E BIAS
            for (int k = 0; k < numOutputs; k++){
                outputLayerBias[k] += deltaOutput[k] * lr;
                for (int j = 0; j < numHidden; j++){
                    outputWeights[j][k] += hiddenLayer[j] * deltaOutput[k] * lr;
                }
            }

            for (int j = 0; j < numHidden; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * lr; // Corretto il '+' in '*'
                for (int i = 0; i < numInputs; i++) {
                    hiddenWeights[i][j] += inputs[p][i] * deltaHidden[j] * lr;
                }
            }
        }
    }

    // TEST FINALE
    printf("Risultati dell'addestramento della rete neurale:\n");

    for (int p = 0; p < 4; p++) {
        for (int j = 0; j < numHidden; j++) {
            double activation = hiddenLayerBias[j];
            for (int i = 0; i < numInputs; i++) {
                activation += inputs[p][i] * hiddenWeights[i][j];
            }
            hiddenLayer[j] = sigmoid(activation);
        }

        for (int k = 0; k < numOutputs; k++) {
            double activation = outputLayerBias[k];
            for (int j = 0; j < numHidden; j++) {
                activation += hiddenLayer[j] * outputWeights[j][k];
            }
            outputLayer[k] = sigmoid(activation);
        }

        // Parentesi e formattazione corretta
        printf("Input: %d %d -> Output predetto: %f (Target: %.0f)\n",
            (int)inputs[p][0], (int)inputs[p][1], outputLayer[0], targets[p]);
    }

    return 0;
}