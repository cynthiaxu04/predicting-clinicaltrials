![alt text][def]

## Problem & Motivation
Clinical trials are pivotal for the development of medical, surgical, or behavioral interventions, yet their long duration—often exceeding 15 years—creates critical bottlenecks in the healthcare innovation pipeline. The capacity to accurately forecast the duration of these trials is vital for enhancing operational efficiency, reducing expenditures, and expediting the delivery of potentially life-saving treatments to patients.

Our product targets Phase 3 oncology trials, a crucial phase in cancer drug development, to significantly enhance resource optimization and cost reduction. Taking in only initial protocol information to generate the predicted duration, our model will not only improve the accuracy of trial duration estimates but also support clinical research organizations in streamlining their operations and allocating resources more effectively.

## Data & Approach

## MVP
Our web API predicts a time interval based on information from the study protocol for Phase 3 oncology trials. Some examples of study information include:
- number of enrolled participants
- number of participant exclusion criteria
- disease category
- number of study sites
- etc.

See ```/trial_app/README.md``` for full instructions on how to build and launch the app.

## Model
Our MVP uses a Random Forest Classifier trained on publicly available data from ClinicalTrials.gov. The model accuracy with 3 bins is 0.602.

See ```/model/README.md``` for details on model training.  
See ```/data/README.md``` for full details on data and feature engineering.


[def]: /figures/ClinicalAI.png