# Training params
TRAIN_PARAMS = {
    "num_epochs": 5,
    "lr": 2e-5,
    "pos_samples_weight": 0.5,
    "train_batch_size": 32,
    "val_batch_size": 512,
    "hidden_dim": 769,
    "pad_length": 256,
    "num_classes": 20
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

