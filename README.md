                                  Breast Cancer Classification using VGG16 + Monarch Butterfly Optimization (MBO)


&nbsp;                                                                              **Project Overview**



This repository implements a complete machine learning pipeline for classifying histopathological breast cancer images (benign vs. malignant).



The pipeline uses: - Deep feature extraction with pretrained VGG16 - Feature selection using Monarch Butterfly Optimization (MBO) - Fitness evaluation with Random Forest - Final classification using Support Vector Classifier (SVC)



&nbsp;                                                                                  **Dataset**



Download the dataset ZIP file from: https://data.mendeley.com/public-files/datasets/jxwvdwhpc2/files/f7d558f5-db9c-4e6a-9245-4967a7f36e56/file\_downloaded



Steps: 1. Download the ZIP file. 

&nbsp; 2. Extract it locally. 

&nbsp; 3. Place the extracted folder inside: data/raw/



Expected structure after extraction: data/raw/ dataset\_cancer\_v1/classificacao\_binaria/ 40X/ benign/malignant/ 100X/ benign/ malignant/ 200X/ 400X/



&nbsp;                                                                              **Project Structure**



vgg16-mbo-breast-cancer-classification/

│

├── data/

│ └── raw/ ← Put dataset here (not uploaded)

│

├── models/ ← Saved trained model + features

│

├── outputs/ ← Optional visual outputs

│

├── src/

│ ├── preprocess.py # Merges \& splits dataset

│ ├── feature\_extraction.py # VGG16 deep feature extraction

│ ├── mbo.py # Monarch Butterfly Optimization

│ ├── classifier.py # SVM training + evaluation

│ └── main.py # Full pipeline execution

│

├── requirements.txt

├── README.md

└── .gitignore



&nbsp;                                                                                   **Installation**



Clone the repository: git clone cd vgg16-mbo-breast-cancer-classification



Create virtual environment (recommended): Windows: python -m venv venv venv



Install dependencies: pip install -r requirements.txt



&nbsp;                                                                                **Running the Pipeline**



After placing dataset inside data/raw/:



python src/main.py



Pipeline Steps: 1. Merge magnification folders 

&nbsp;		    2. Train-test split (80/20) 

&nbsp;		    3. Extract deep features using VGG16 

&nbsp;		    4. Run MBO feature selection 

&nbsp;		    5. Train final SVC classifier 

&nbsp;		    6. Print evaluation metrics 

&nbsp;		    7.Save trained model and selected features



&nbsp;                                                                                   **Saved Artifacts**



models/ svm\_model.pkl selected\_features.pkl



These can be reloaded using:import joblib clf = joblib.load(“models/svm\_model.pkl”) 

&nbsp;			    features = joblib.load(“models/selected\_features.pkl”)



Evaluation Metrics



\-   Test Accuracy

\-   Confusion Matrix

\-   Precision, Recall, F1-score



librarys/packages Used



\-   Python

\-   TensorFlow / Keras

\-   Scikit-learn

\-   NumPy

\-   Pandas



Future Improvements



\-   Hyperparameter tuning

\-   Cross-validation

\-   ROC/AUC reporting

\-   Comparison with other optimization algorithms

