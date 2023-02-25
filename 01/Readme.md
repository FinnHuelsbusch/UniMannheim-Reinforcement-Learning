# Explenation

## Task 4

Approaches: 

- Random 
  - total_scores: [2184, 2200, 2278, 2167, 2277, 2207, 2287]
- Explore for n iterations to gain Information. After these n iterations follow the action with the highest reward.
  Shown with 1000
  - total_scores: [6468, 6746, 6593, 6631, 6772, 6646, 6825]
- Similar to simulated Annealing
  - total_scores_median: [3422, 3589, 3293, 3310, 3346, 3335, 3417]
    - probability_to_explore = 1
    - discount_factor = 0.9999
  - [7494, 7871, 7593, 7487, 7508, 7669, 7839]
    - probability_to_explore = 1
    - discount_factor = 0.99
  - [7497, 7589, 7391, 7701, 7527, 7423, 7843]
    - probability_to_explore = 0.9
    - discount_factor = 0.99
  - [7861, 3305, 7755, 7579, 7872, 3724, 2537]
    - probability_to_explore = 0.9
    - discount_factor = 0.9
  - [3806, 7945, 7460, 7574, 7864, 3726, 7923]
    - probability_to_explore = 0.9
    - discount_factor = 0.95
  - [7826, 7640, 7411, 7810, 7853, 7745, 7610]
    - probability_to_explore = 0.9
    - discount_factor = 0.975

# Notizen

## Exercise 1 
Es gibt immer eine Aufgabenblatt und mehrere Python files. 
Auf dem ersten Aufgabenblatt stehen generelle Angaben. Bitte daran halten. Abgabe per mail. Es gibt manchmal auch theorie Fragen. Diese immer als PDF beantworten. 
Wir sollten uns an die generelle Struktur in den Dateien halten. 
Es sind vorgegebenen Bibliotheken. Aber die nur dann verwenden wenn nicht explizit anders verlangt. Keine Bibliotheken verwenden wenn nicht anders verlangt. Erst recht nicht dann, wenn die Aufgabe ist etwas theoretisch nach zu bauen 

### Aufgabe 1 Policy Evaluation. 
An die Reihenfolge halten. 
Use numpy for the inversion of the matrix 
Eine zwei mal acht matrix. Zwei für up und down und acht für die acht möglichen stadien. Im py file r 
### Aufgabe 2 
Erst nach der nächsten Vorlesung 

### Aufgabe 3 
Erst nach der zweiten Vorlesung 

### Aufgabe 4 
Wird nicht in der Vorlesung beantwortet. 
Ist ein MDP, der immer wieder in den selben Zustand zurück kommt. 
Hier ist der reward stockastic. Je nach Aktion. Wir kennen die Stochastic aber nicht. 
Es gibt keine Abschreibung 
10 gaus verteilungen mit unterschiedlicher varianz und mean. 

Getestet wird mit einem anderen Banditen. 
Algorithmus sollte sich an neuen Banditen anpassen können. 

