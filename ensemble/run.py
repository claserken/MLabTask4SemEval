from config import MODELS_DIR, HUMAN_VALUES, DATA_FNAMES
from BERT.src.config import TRAIN_PARAMS
from ensemble import EnsembledModel, EnsembleStatistics
from model import BERTPredictor, GlovePredictor
import numpy as np

bert_16 = BERTPredictor(MODELS_DIR['bert'] + 'bert_b16-w40-e5', TRAIN_PARAMS)
bert_64 = BERTPredictor(MODELS_DIR['bert'] + 'bert_b64-w50-e9', TRAIN_PARAMS)
glove_predictors = []
glove_params = {
    'sentence_cols': ['Conclusion', 'Premise'],
    'words_to_remove': ['the', 'a', 'an', 'of']
}

for human_value in HUMAN_VALUES:
    clf_fname = MODELS_DIR['glove'] + human_value + '.joblib'
    glove_predictor = GlovePredictor(clf_fname, glove_params)
    glove_predictors.append(glove_predictor)

ensemble = EnsembledModel([[bert_16], [bert_64], glove_predictors], HUMAN_VALUES)
ensemble.train_log_regs(DATA_FNAMES['train_arguments'], DATA_FNAMES['train_labels'])

stats = EnsembleStatistics(ensemble)
f1_scores = stats.f1_score(DATA_FNAMES['train_arguments'], DATA_FNAMES['train_labels'], verbose=True)
mean_f1_score = np.mean(f1_scores)
print(f'Average F1 score: {mean_f1_score}')

# Test set evaluation
# test_dataset_frame = pd.read_csv(DATA_FNAMES['test_arguments'], sep='\t')
# labels_training = pd.read_csv(DATA_FNAMES['train_labels'], sep='\t')

# test_preds = ensemble.ensemble_predict(DATA_FNAMES['test_arguments'])

# final_test_results = pd.DataFrame(test_preds)
# final_test_results.insert(loc=0, column='Argument ID', value=np.nan)
# final_test_results['Argument ID'] = test_dataset_frame['Argument ID']
# final_test_results.columns=labels_training.columns.values
# final_test_results.to_csv('argument_test_preds_bert.tsv', sep="\t", index=False)


    








