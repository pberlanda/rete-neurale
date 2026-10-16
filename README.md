# progetto rete neurale #
## 4INA: implementazione rete neurale a due livelli ##

Una rete neurale artificiale è un modello computazionale ispirato alla struttura del cervello umano. è composta da **nodi (i neuroni)** organizzati in **livelli (layer)**:
1. **Input layer**: riceve i dati in ingresso
2. **Hidden layer**: elabora le informazioni combinando **ingressi**, **pesi (weights)** e un *termine errore* chiamato **bias**, per poi passare il risultato attraverso una **funzione di attivazione** (la Sigmoide).
3. **Output layer** fornisce la predizione finale

nel processo di addestramente la rete calcola un errore (differenza tra la risposta attesa e la risposta ottenuta) e aggiorna i pesi usando l'algoritmo di *back propagation* e la discesa del gradiente (??? che è??)

## Esempio pratico: la porta logica XOR ##

Implementiamo una rete neurale a 2 ingressi, 2 neuroni nascosti e 1 neurone di output per apprendere la funzione logica XOR, che non è risolvibile linearmente (in che senso? che significa risolvere la funzione XOR? devo risolverla? non è una primitiva? e poi che vuol dire che non è risolvibile linearmente???)
