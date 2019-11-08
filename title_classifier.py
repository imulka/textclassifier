'''
Title classification tool that uses textstat for extracting statistics about the target text and
xgboost to perform learning and prediction.

How to run the tool:
First you need to train the machine learning model using pre-labeled data. This data must be in the form of a CSV file
containing 2 columns - title and label. You can use custom names for those columns.
The titles column must contain both good and bad titles in more or less equal amounts.
The labels column must contain one of the two label words: good or bad.

The tool accepts the follwing arguments:
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
pandas
xgboost
textstat
sklearn
'''
import pickle
import textstat
import argparse
import pandas as pd
from xgboost import XGBClassifier

# Handle command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument('-mode', required=True)
parser.add_argument('-title_column', required=False)
parser.add_argument('-load_data_from', required=False)
parser.add_argument('-label_column', required=False)
parser.add_argument('-save_model_to', required=False)
parser.add_argument('-load_model_from', required=False)
parser.add_argument('-save_predictions_to', required=False)
parser.add_argument('-single_prediction', required=False, default='', type=str)
args = parser.parse_args()

if __name__ == '__main__':
    # Statistics that will be calculated for each title.
    stats = [
        'flesch_reading_ease',
        'smog_index',
        'flesch_kincaid_grade',
        'coleman_liau_index',
        'automated_readability_index',
        'dale_chall_readability_score',
        'difficult_words',
        'linsear_write_formula',
        'gunning_fog']

    # Load the data if this is not single prediction.
    if not args.single_prediction:
        if '.xls' in args.load_data_from:
            df = pd.read_excel(args.load_data_from)
        else:
            df = pd.read_csv(args.load_data_from)
        for stat in stats:
            df[stat] = df[args.title_column].apply(getattr(textstat, stat))

# Learning mode.
if args.mode == 'learn':
    X = df[stats]
    Y = df[args.label_column].apply(lambda x: 0 if x.lower() == 'bad' else 1)
    model = XGBClassifier()
    model.fit(X, Y)
    # Save the model to file.
    with open(args.save_model_to, 'wb') as f:
        pickle.dump(model, f)

    print('Model saved to: {}'.format(args.save_model_to))

# Prediction mode.
elif args.mode == 'predict' and not args.single_prediction:
    # Load the pre-trained mode.
    with open(args.load_model_from, 'rb') as f:
        model = pickle.load(f)

    results = []
    for index, row in df.iterrows():
        text = row[args.title_column]
        data = pd.DataFrame({s:[getattr(textstat, s)(text)] for s in stats})
        results.append(model.predict(data))

    # Generate output.
    df.drop(stats, inplace=True, axis=1)
    df['result'] = list(map(lambda x: 'good' if x[0] else 'bad', results))
    df.to_csv(args.save_predictions_to, index=False)

    print('Predictions saved to: {}'.format(args.save_predictions_to))

# Single prediction mode.
elif args.mode == 'predict' and args.single_prediction:
    # Load the pre-trained mode.
    with open(args.load_model_from, 'rb') as f:
        model = pickle.load(f)

    # Make single prediction.
    text = args.single_prediction
    data = pd.DataFrame({s: [getattr(textstat, s)(text)] for s in stats})
    result = 'good' if model.predict(data)[0] else 'bad'
    print('Prediction: {}'.format(result))
