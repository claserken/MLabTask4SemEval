TRAIN_PREDS_DIR = {
    'bert': './BERT/train_predictions/',
    'glove': './glove/train_predictions/'
}

MODELS_DIR = {
    'bert': './BERT/saved_models/',
    'glove': './glove/saved_models/'
}

HUMAN_VALUES = ['Self-direction: thought', 'Self-direction: action', 'Stimulation',
                'Hedonism', 'Achievement', 'Power: dominance', 'Power: resources',
                'Face', 'Security: personal', 'Security: societal', 'Tradition',
                'Conformity: rules', 'Conformity: interpersonal', 'Humility',
                'Benevolence: caring', 'Benevolence: dependability',
                'Universalism: concern', 'Universalism: nature',
                'Universalism: tolerance', 'Universalism: objectivity']

# File locations for data
DATA_URI = "./data/"

TRAINING_ARGUMENTS_FNAME = DATA_URI + "arguments-training.tsv"
TRAINING_LABELS_FNAME = DATA_URI + "labels-training.tsv"

VALIDATION_ARGUMENTS_FNAME = DATA_URI + "arguments-validation.tsv"
VALIDATION_LABELS_FNAME = DATA_URI + "labels-validation.tsv"

DATA_FNAMES = {
    "train_arguments": TRAINING_ARGUMENTS_FNAME,
    "train_labels": TRAINING_LABELS_FNAME,
    "valid_arguments": VALIDATION_ARGUMENTS_FNAME,
    "valid_labels": VALIDATION_LABELS_FNAME
}