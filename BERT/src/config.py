# Training params
NUM_EPOCHS = 5
LR = 2e-5

TRAIN_PARAMS = {
    "num_epochs": NUM_EPOCHS,
    "lr": LR
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

