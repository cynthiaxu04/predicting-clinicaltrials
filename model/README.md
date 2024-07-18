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
python train_model.py <model type>
```
Where ```<model type>``` is one of the CLI arguments listed above.

If you are training a model for production, use the ```--prod``` flag to save a copy of the model ```pkl``` file and the associated metadata in the ```trial_app/backend``` folder, e.g.
```
python train_model.py <model type> --prod
```

This is what the script does:
1. Reads in the cleaned training dataset produced by the ```processing.py``` script.
    1. This script assumes that training dataset is ```../data/cleaned_data_train.csv```.
2. Splits the data into train & test for model training.
3. Trains the selected classification model.
4. Saves the trained model as a ```pkl``` file.
5. Saves the columns used to train the model in a file ```model_columns_<model type>.txt```. These columns are picked based on correlation values, so they may change. To predict on the test set, the exact same columns must be used.
6. Saves the model metrics (i.e., accuracy, mean accuracy from k-fold cross validation, precision, R-squaread, mean squared error, and mean absolute error) to the file ```modeling_results.json```.

## ```get_test_results.py``` for evaluating the model

The purpose of this script is to evaluate the trained model on the test set ```cleaned_data_test.csv```.

To run this script, ensure you are in ```model``` directory. Then, run the command in terminal:
```
python get_test_results.py <model type>
```
Where ```<model type>``` is one of the CLI arguments listed above.

This is what the script does:
1. Reads in the cleaned testing dataset produced by the ```processing.py``` script.
    1. This script assumes the test dataset is ```../data/cleaned_data_test.csv```
2. Reads in the saved model ```pkl``` file produced by the ```train_model.py``` script.
3. Predicts results for the test set.
4. Outputs the results of the model evaluation in ```modeling_test_results.json```.