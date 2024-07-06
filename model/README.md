# Model Training

## ```train_model.py``` for training and saving a model

The purpose of this script is to train and save a model in a ```pkl``` file. The current model options are:
| Model | CLI Argument|
| -------- | -------- |
|```lightgbm.LGBMClassifier()```|```lgb```|
|```xgboost.XGBClassifer()```|```xgb```|
|```RandomForestClassifier()```|```rf```|

To run this script, ensure you are in ```model``` directory. Then, run the command in terminal:
```
python train_model.py ../data/cleaned_data_train.csv <model type>
```
Where ```<model type>``` is one of the CLI arguments listed above.

This is what the script does:
1. Reads in the cleaned training dataset produced by the ```processing.py``` script.
2. Splits the data into train & test for model training.
3. Trains the selected classification model.
4. [Optional] Saves the trained model as a ```pkl``` file.
5. Saves the model metrics (i.e., accuracy, mean accuracy from k-fold cross validation, precision, R-squaread, mean squared error, and mean absolute error) to the file ```modeling_results.json```.