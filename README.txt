Title classification tool that uses textstat for extracting statistics about the target text and
xgboost to perform learning and prediction.

How to run the tool:
First you need to train the machine learning model using pre-labeled data. This data must be in the form of a CSV file
containing 2 columns - title and label. You can use custom names for those columns.
The titles column must contain both good and bad titles in more or less equal amounts.
The labels column must contain one of the two label words: good or bad.

The tool accepts the following arguments:
-mode -> Can be learn or predict
-title_column -> The column in the input data file that contains the titles. (project_titles)
-load_data_from -> The filename of the input data. (data.csv)
-label_column -> The column in the input data file that contains the labels (good or bad).
-save_model_to -> The filename of the saved model (my_model.pkl)
-load_model_from -> The filename of the pre-trained model to load for predictions (my_model.pkl)
-save_predictions_to -> The filename of the CSV file that will contain the predictions. (my_predictions.csv)
-single_prediction -> Generate prediction for single title. (This title looks good so it will trick the computer.)

Examples:
1. Use CSV file called data.csv using a column called project_title and good_bad as input and save the trained model to a file called trained_model.pkl
    python title_classifier.py -mode learn -load_data_from data.csv -title_column project_title -label_column good_bad -save_model_to trained_model.pkl

2. Use CSV file called data.csv using a column called project_title and good_bad as input. Load the pre-trained model from a file called trained_model.pkl and save predictions to a CSV file called my_predictions.csv
    python title_classifier.py -mode predict -load_data_from data.csv -title_column project_title -label_column good_bad -load_model_from trained_model.pkl -save_predictions_to my_predictions.csv

3. Load the pre-trained model from a file called "trained_model.pkl" and use to predict the title "Randomness in big numbers and other elements"
    python title_classifier.py -mode predict -load_model_from trained_model.pkl -single_prediction "Randomness in big numbers and other elements"

Requirements:
pandas, sklearn, xgboost, textstat, xlrd