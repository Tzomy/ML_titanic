# 🚢 Titanic - Machine Learning from Disaster (Kaggle Competition)

Questo repository contiene la mia soluzione per la celebre competizione di Kaggle **[Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)**. L'obiettivo del progetto è prevedere quali passeggeri sono sopravvissuti al naufragio del Titanic applicando tecniche di *Data Preprocessing*, *Feature Engineering*, *Machine Learning* tradizionale e *Deep Learning* (Reti Neurali).

🏆 **Piazzamento in Leaderboard:** Top 30%

---

## 📋 Descrizione del Problema e del Dataset

Il naufragio del Titanic è uno dei più famosi della storia. Il 15 aprile 1912, durante il suo viaggio inaugurale, il Titanic affondò dopo essere entrato in collisione con un iceberg. Purtroppo non c'erano abbastanza scialuppe di salvataggio per tutti i passeggeri a bordo, causando la morte di 1502 persone su 2224 tra passeggeri ed equipaggio. Sebbene ci fosse una certa dose di fortuna nel sopravvivere, sembra che alcuni gruppi di persone avessero maggiori probabilità di sopravvivenza rispetto ad altri (ad esempio donne, bambini e passeggeri di prima classe).

Il dataset fornito è suddiviso in `train.csv` (891 righe) e `test.csv` (418 righe) e comprende le seguenti feature principali:
* **`Survived`**: Sopravvivenza ($0 = \text{No}$, $1 = \text{Sì}$) — *Variabile Target*
* **`Pclass`**: Classe del biglietto ($1 = \text{1ª classe}$, $2 = \text{2ª classe}$, $3 = \text{3ª classe}$)
* **`Sex`**: Sesso del passeggero
* **`Age`**: Età in anni (presenta valori mancanti)
* **`SibSp`**: Numero di fratelli/sorelle o coniugi a bordo
* **`Parch`**: Numero di genitori o figli a bordo
* **`Ticket`**: Numero del biglietto
* **`Fare`**: Tariffa del passeggero
* **`Cabin`**: Numero della cabina
* **`Embarked`**: Porto di imbarco ($C = \text{Cherbourg}$, $Q = \text{Queenstown}$, $S = \text{Southampton}$)

---

## 🛠️ Data Preprocessing & Feature Engineering

Per preparare i dati all'addestramento dei modelli, sono state eseguite le seguenti operazioni di pulizia e trasformazione:

1. **Codifica Variabili Categoriche:**
   * La colonna `Sex` è stata convertita in formato binario numerico (`male` = $1$, `female` = $0$).
   * Le colonne testuali o con valori misti (`Cabin`, `Embarked`) sono state convertite in stringhe per evitare errori di tipo durante l'analisi esplorativa.

2. **Gestione dei Valori Mancanti e Binning dell'Età (`Age`):**
   * L'analisi esplorativa ha evidenziato valori nulli principalmente nella colonna `Age`.
   * Invece di procedere con una semplice imputazione tramite media o mediana, i valori mancanti sono stati riempiti con $-1$ e la variabile continua `Age` è stata trasformata in una **variabile categorica ordinale** (`Age_categories`) suddivisa nelle seguenti fasce:
     * **`Missing`**: $[-1, 0]$
     * **`Infant`**: $(0, 5]$
     * **`Child`**: $(5, 12]$
     * **`Teenager`**: $(12, 18]$
     * **`Young Adult`**: $(18, 35]$
     * **`Adult`**: $(35, 60]$
     * **`Senior`**: $(60, 100]$

3. **One-Hot Encoding (Variabili Dummy):**
   * Sono state generate variabili *dummy* per `Pclass` e per `Age_categories`.
   * Il set finale di **11 feature** selezionate per l'input dei modelli è composto da:
     `['Sex', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Age_categories_Missing', 'Age_categories_Infant', 'Age_categories_Child', 'Age_categories_Teenager', 'Age_categories_Young Adult', 'Age_categories_Adult', 'Age_categories_Senior']`

4. **Train/Test Split:**
   * Il dataset di training è stato suddiviso in **Training Set (80%)** e **Validation/Test Set (20%)** con `random_state=0` per valutare le performance dei modelli prima della submission finale.

---

## 📊 Exploratory Data Analysis (EDA)

Nel notebook sono state realizzate diverse visualizzazioni tramite `matplotlib` e `seaborn` per comprendere le correlazioni tra le feature:
* **Pivot Table & Bar Plot:** Analisi del tasso di sopravvivenza medio per ciascuna fascia d'età (`Age_categories`).
* **Distribuzioni per Classe (`Pclass`):** Istogrammi di densità di probabilità per confrontare come le varie feature (età, tariffa, sopravvivenza, ecc.) si distribuiscono tra 1ª, 2ª e 3ª classe.
* **Matrice di Correlazione:** Heatmap di correlazione tra le variabili numeriche principali (`Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`).

---

## 🤖 Modelli di Machine Learning (Scikit-Learn)

Sono stati addestrati e confrontati quattro diversi algoritmi di classificazione sul Validation Set:

| Modello | Dettagli / Iperparametri Principali | Accuracy | Log Loss |
| :--- | :--- | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | `n_neighbors=3`, `weights='uniform'`, `algorithm='kd_tree'` | **83.24%** | $6.04$ |
| **Support Vector Machine (SVM)** | `SVC()` (kernel RBF di default) | **82.68%** | $6.24$ |
| **Gradient Boosting Classifier** | `loss='exponential'`, `criterion='squared_error'`, `max_depth=2` | **82.12%** | $6.44$ |
| **Naive Bayes** | `MultinomialNB(force_alpha=True, fit_prior=True)` | **81.01%** | $6.85$ |
| **Logistic Regression** | `penalty='l2'`, `solver='liblinear'` | **81.01%** | $6.85$ |

> 💡 **Nota:** Il modello **KNN** è risultato il migliore tra gli algoritmi classici di Machine Learning con un'accuratezza superiore all'**83%** sul set di validazione.

---

## 🧠 Deep Learning & Tuning di Reti Neurali (TensorFlow / Keras)

Per migliorare ulteriormente la capacità predittiva e ridurre la *Binary Crossentropy Loss*, sono state progettate diverse architetture di **Reti Neurali Dense (Sequential)** utilizzando `TensorFlow` e l'ottimizzatore `Adam`.

### Architetture Testate
1. **Rete a 2 Layer Nascosti:** 2 strati `Dense` intervallati da strati di `Dropout` e uno strato di output con attivazione `sigmoid`.
2. **Rete Profonda a 5 Layer Nascosti:** 5 strati `Dense` intervallati da `Dropout` e output `sigmoid`.

### Grid Search sugli Iperparametri
È stata effettuata una ricerca sistematica (100 epoche per configurazione, con `validation_split=0.2`) variando:
* **Numero di Nodi per Layer:** `[16, 32]`
* **Probabilità di Dropout:** `[0, 0.2]`
* **Learning Rate (`lr`):** `[0.1, 0.005, 0.001]`
* **Batch Size:** `[32, 64, 128]`
* **Funzioni di Attivazione testate:** `relu`, `sigmoid`, `gelu`, `elu`, `hard_sigmoid`, `mish`, `tanh`, `exponential`, `linear`, `selu`, `softmax`, `softplus`, `softsign`.

### Risultati Principali delle Reti Neurali
* Le funzioni di attivazione come **`gelu`**, **`elu`**, **`selu`** e **`softsign`** hanno permesso di abbattere drasticamente la *validation loss* (scendendo fino a **$\sim 0.369 - 0.377$** rispetto al $> 6.0$ dei modelli classici).
* In termini di accuratezza sul test set interno, configurazioni con **`hard_sigmoid`** e **`softsign`** hanno raggiunto picchi dell'**83.24%**, eguagliando il KNN ma con una calibrazione delle probabilità (loss) nettamente migliore.

---

## 📁 File di Submission Generati

Alla fine del notebook vengono esportati i file `.csv` pronti per l'upload su Kaggle:
* `titanic_submission_knn.csv`: Predizioni generate con il modello K-Nearest Neighbors.
* `titanic_submission_1.csv` / `titanic_submission.csv`: Predizioni ottenute dai modelli neurali con la minor *validation loss* (`least_loss_model`).
* `titanic_submission_champ.csv`: Predizioni generate dal modello neurale con la più alta accuratezza in fase di validazione (`highest_acc_model`).

---

## 🚀 Come Eseguire il Progetto

1. Clona il repository contenente i dati:
   ```bash
   git clone https://github.com/Tzomy/ML_titanic
   ```
2. Installa le dipendenze necessarie:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn tensorflow
   ```
3. Apri ed esegui il Jupyter Notebook (su Google Colab o in locale) per riprodurre i grafici, l'addestramento dei modelli e generare i file `.csv` per la submission su Kaggle.
