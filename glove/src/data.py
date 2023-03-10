from torchtext.vocab import GloVe
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import re

class DataHandler(ABC):
    def filter_words(self, df, col_names, words_to_remove):
        filtered_dict = {}
        for idx in range(len(df)):
            new_arg_entry = {}
            for col in col_names:
                line = df[col][idx].lower()
                rem_funcs = re.findall(r'(?:\w+)', line, flags=re.UNICODE)
                new_sent = []
                for word in rem_funcs:
                    if word not in words_to_remove:
                        new_sent.append(word)
                new_arg_entry[col] = new_sent
            
            filtered_dict[df["Argument ID"][idx]] = new_arg_entry
        return filtered_dict
    
    @abstractmethod
    def transform_data(self, df, sentence_cols, words_to_remove):
        pass

    def transform_dataset_from_file(self, data_fname, sentence_cols, words_to_remove):
        dataset = pd.read_table(data_fname)
        stances = np.array(dataset['Stance'] == 'in favor of', dtype=int).reshape(-1, 1)

        dataset = self.transform_data(dataset, sentence_cols, words_to_remove)
        dataset = np.concatenate((stances, dataset), axis=1)
        return dataset
    
class GloveEmbedder(DataHandler):
    def __init__(self, glove_dim):
        self.global_vectors = GloVe(name="6B", dim=glove_dim)

    def embed_sentences(self, data_dict, col_names):
        embedded_sentences = {}
        for arg_id in data_dict:
            temp = {}
            for col in col_names:
                sent = data_dict[arg_id][col]
                embedded_sent = np.zeros(100)

                for word in sent:
                    embedded_sent += self.global_vectors.get_vecs_by_tokens(word).numpy()
                embedded_sent /= len(embedded_sent)
                temp[col] = embedded_sent
            embedded_sentences[arg_id] = temp
        return embedded_sentences

    def transform_data(self, df, sentence_cols, words_to_remove):
        data = self.filter_words(df, sentence_cols, words_to_remove)
        data = self.embed_sentences(data, sentence_cols)
        for key in data:
            data[key] = data[key]["Conclusion"] + data[key]["Premise"] 
        data = pd.DataFrame(data).values.T
        return data
    

    




