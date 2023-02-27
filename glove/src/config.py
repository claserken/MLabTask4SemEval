# Random seed
SEED = 0

# Training params
TRAIN_PARAMS = {
    "glove_dim": 100
}

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