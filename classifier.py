import datetime
import io
import os
import numpy as np
import pandas as pd
import storage
from   objdict                      import ObjDict
from   sklearn.model_selection      import train_test_split
from   sklearn                      import metrics
from   keras.preprocessing.text     import Tokenizer
from   keras.preprocessing.sequence import pad_sequences
from   keras.layers                 import Dense, Flatten
from   keras.layers                 import Conv1D, MaxPooling1D, Embedding
from   keras.models                 import Sequential
from   keras.callbacks              import ModelCheckpoint

def score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * precision * recall / (precision + recall)

    
#-#-#-#-#----CLASS----#-#-#-#-#
class TrendClassifier:
    def __init__(self):
        pass
        
    def build_model( self, 
                     exp_name, 
                     model_creator = None, 
                     model         = None, 
                     **args):
        return (None, 0)
        
    def classify(self, 
                 exp_name, 
                 model):
        pass 

        
        
#-#-#-#-#----CLASS----#-#-#-#-#
class CNNTrendClassifier(TrendClassifier):
    def __init__(self, 
                 configuration, 
                 module):
        print("C N N    T R E N D     C L A S S I F I E R")
        self.config           = configuration
        self.embedding_index  = None
        self.embedding_matrix = None
        self.tokenizer        = None
        self.max_input_len    = 0
        self.sample_sequences = None
        self.sample_data      = None
        self.input_data       = None
        self.input_sequences  = None
        self.vocabulary_size  = 0
        self.classes          = None
        self.module           = configuration.modules[module]
        self.trends           = self.module.trends
        self.s3               = storage.create_storage(self.config)
        self.__load_sample_data()
        self.__initialize_word_vector()
        self.__build_embedding_matrix()
        self.__load_input_data()

    def __load_sample_data(self, 
                           filename = None):
        filename              = filename or self.config.paths.sanitized_data
 
        self.sample_data, input_sequences, self.tokenizer = self.__load_token_file( filename, 
                                                                                    balance_dataset = True)
        self.max_input_len    = max([len(seq) 
                                            for seq in input_sequences ])
        self.sample_sequences = pad_sequences( input_sequences, 
                                               maxlen     = self.max_input_len, 
                                               padding    = 'post', 
                                               truncating = 'post'
                                              )
        self.classes = self.sample_data.columns[2:].values

    def __load_input_data(self, filename = None):
        filename                        = filename or self.config.paths.unlabeled_data
        self.input_data, sequences, _   = self.__load_token_file( filename, self.tokenizer)
        self.input_sequences            = pad_sequences( sequences, 
                                                        maxlen      = self.max_input_len, 
                                                        padding     = 'post', 
                                                        truncating  = 'post')
          
    def __initialize_word_vector(self, 
                                 word_vector_file = None):
        word_vector_file              = word_vector_file or self.config.paths.word_vector_data        
        print("LOADING WORD VECTORS...{}".format(word_vector_file))
        embedding_index               = {}
        with open(word_vector_file, encoding = 'UTF8') as file: 
            for line in file:
                values                = line.split()
                word                  = str.lower(values[0])
                coefs                 = np.asarray(values[1:], dtype='float32')
                embedding_index[word] = coefs
        print("\t{} word vectors loaded.".format(len(embedding_index)))
        self.embedding_index          = embedding_index        
        return embedding_index

    def __build_embedding_matrix(self, 
                                 vocabulary_size = None, 
                                 embedding_dim   = None):
        vocabulary_size       = vocabulary_size or self.config.runtime.max_vocabulary_size
        embedding_dim         = embedding_dim   or self.config.runtime.embedding_dim
        word_index            = self.tokenizer.word_index 
        self.vocabulary_size  = min( vocabulary_size, 
                                     len(word_index) + 1)
        embedding_matrix      = np.zeros((self.vocabulary_size, 
                                          embedding_dim))        
        for word, i in word_index.items():
            if i >= vocabulary_size:
                continue
            embedding_vector        = self.embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        self.embedding_matrix = embedding_matrix

    def __load_token_file(self, 
                          filename, 
                          tokenizer       = None, 
                          token_column    = 'summary', 
                          balance_dataset = False):
        """
        Load the token file to the memory. 
            :param filename: the token file name
            :param tokenizer: the tokenizer, if None a new tokenizer will be created.
        """
        print('LOADING TOKEN FILE  {0}...'.format(filename))
        data     = self.s3.read_csv(filename)

        if balance_dataset:
            data = self._balancingDataset(data)

        if tokenizer is None:
            tokenizer        = Tokenizer(num_words = self.vocabulary_size)
            tokenizer.fit_on_texts(data[token_column])
        
        sample_sequences     = tokenizer.texts_to_sequences(data[token_column])
        return data, sample_sequences, tokenizer

    def _balancingDataset(self, 
                          data: pd.DataFrame) -> pd.DataFrame:
        # counting support elements per trends
        print("RECALCULATE VALUES FOR TREND 'OTHER'...")
        data['Other']    = data[self.module.trends].apply(lambda x: 0 
                                                                     if sum(x) > 0 
                                                                     else 1, 
                                                                 raw = True, 
                                                                 axis = 1)
        max_count        = 0
        trend_with_other = self.trends + ['Other']
        for trend in trend_with_other:
            count        = data[trend].sum()
            if trend != 'Other' and max_count < count:
                max_count = count
            print("Support values for trend '{0}': {1}".format(trend, count))

        print('Rebalancing samples...')
        ranges        = range(0, len(data))
        other_indexes = [i for i in ranges 
                                if data.iloc[i].Other == 1
                         ]
        selector      = np.random.random_sample(len(other_indexes))
        threshold     = max_count*1.2/len(other_indexes)
        to_removed_other_indexes = [ val for idx, 
                                     val in enumerate(other_indexes) 
                                         if selector[idx] >= threshold
                                    ]
        subrange      = [x for x in ranges 
                                if not x in to_removed_other_indexes
                        ]
        data          = data.iloc[subrange]
        for trend in trend_with_other:
            print("Support values for trend '{0}': {1}".format(trend, 
                                                               data[trend].sum())
                  )
        return data

    def get_vocabulary_size(self):
        return min( self.config.runtime.max_vocabulary_size, 
                    len(self.tokenizer.word_index) + 1
                   )

    def train_model(self, 
                    name, 
                    model, 
                    train_data, train_label):
        model_file = self.config.paths.checkpoint_file.format(name)
        if (os.path.isfile(model_file)):
            try:
                model.load_weights(model_file)
                print("Reload weight from previous training: {}".format(model_file))
                if self.config.runtime.train_model == 'auto':
                    return model
            except Exception as e:
                print(str(e))
                print("Traning from scratch, checkpoint = {}".format(model_file))
                pass

        print("TRAINING  CNN  MODEL ...")
        print("Sample size: {}, step-test-size: {}".format( len(train_data), 
                                                            self.config.runtime.step_test_size)
              )
        best_model_saver = ModelCheckpoint( model_file, 
                                            monitor        = 'val_acc', 
                                            verbose        = 1, 
                                            save_best_only = True)
        # HAPPY LEARNING !
        model.fit(train_data, 
                  train_label, 
                  epochs           = self.config.runtime.epoch_count, 
                  verbose          = 0, 
                  validation_split = self.config.runtime.step_test_size,
                  callbacks        = [best_model_saver]
                  )
        model.load_weights(model_file)
        return model
    
    def print_report(self, 
                     classified_values, 
                     actual_values, 
                     classes):
        output_file         = self.config.paths.summary_file
        actual_values       = pd.DataFrame(actual_values,     columns = classes)
        classified_values   = pd.DataFrame(classified_values, columns = classes)
        f                   = open(output_file, 
                                   encoding = 'UTF8', 
                                   mode = 'a+')        
        f.write('-------------------------\n')
        f.write('Test on {}\n'.format(datetime.datetime.now()))
        f.write('Parameters\n\tTest set: {}\n\tEpocs: {}\n'.format(self.config.runtime.test_set_size, 
                                                                   self.config.runtime.epoch_count)
                )
        print('Test set size: {}'.format(len(classified_values)))
        acceptance_threshold = self.module.acceptance_threshold or 0.5
        scores               = []
        for col in classes:
            actual           = [x[0] for x in actual_values[[col]].values]
            classified       = [1 if x[0] >= acceptance_threshold 
                                  else 0 for x in classified_values[[col]].values
                                ]
            precision, recall, score, support = metrics.precision_recall_fscore_support(actual, 
                                                                                        classified
                                                                                        )
            text             = "Accuracy/Precision/Recall of {:<25}: {:6.2f}/{:6.2f}/{:6.2f}, score: {:4.2f}, supports: {}".format(
                                                   col, 
                                                   metrics.accuracy_score(actual, classified), 
                                                   precision[1],
                                                   recall[1],
                                                   score[1],
                                                   support[1])
            scores.append(score[1])
            print(text)
            print(text, 
                  file = f)
        avg_scores     = sum(scores)/len(scores)
        print('Overral score: {}'.format(avg_scores))
        print('Overral score: {}'.format(avg_scores), file=f)
        f.close()
        return avg_scores
   
    def build_model(self, 
                    exp_name, 
                    model_creator = None, 
                    model         = None, 
                    **args):
        if (model is None) and (model_creator is None):
            raise Exception('either model or model_creator should not be None')

        print("RUNNING EXPERIMENT ON : {0}".format(exp_name))
        label_columns          = self.sample_data.columns[2:].values
        selected_labels_data   = np.asarray(self.sample_data[label_columns])

        model = model or model_creator(self, **args)
        
        selector, classified_values, _, score = self.__validate(exp_name, 
                                                                model, 
                                                                self.sample_sequences, 
                                                                selected_labels_data
                                                                )
        self.export_detail_classification( self.config.paths.validation_output_data.format(exp_name), 
                                           self.sample_data.iloc[selector <= self.config.runtime.test_set_size], 
                                           classified_values, 
                                           trend_column_prefix = 'classified_'
                                          )
        return model, score

    def classify(self, 
                 exp_name, 
                 model):
        """
        :param model: the model to perform classification over input data
        :param exp_name: the experiment name
        """
        print("RUNNING CLASSIFICATION ON :  {0}".format(exp_name))
        classified_values = pd.DataFrame( model.predict(self.input_sequences, 
                                                        verbose = 1), 
                                          columns = self.classes
                                         )
        self.export_detail_classification(self.config.paths.classification_output_data.format(exp_name), 
                                          self.input_data, 
                                          classified_values
                                         )
        return classified_values

    def __validate(self, name, model, data, labels):
        """
        Run the train/test validation of a model on given classes, data, and labels

        :param model: the model to do validation
        :param data: the array of padded sequences, used as input data
        :param labels: the array of actual classes of the padded sequences
        """
        if self.config.runtime.seed > 0:
            np.random.seed(self.config.runtime.seed)

        selector  = np.random.random(len(data))
        test_size = self.config.runtime.test_set_size

        self.train_model(name, 
                         model, 
                         data[selector > test_size], 
                         labels[selector > test_size]
                         )
        classified_values = model.predict(data[selector <= test_size], 
                                          verbose = 1
                                          )
        actual_values     = pd.DataFrame(labels[selector <= test_size], 
                                          columns = self.classes
                                         )
        classified_values = pd.DataFrame(classified_values, 
                                         columns = self.classes
                                         )
        score             = self.print_report(classified_values, 
                                              actual_values, 
                                              self.classes
                                              )
        return selector, classified_values, actual_values, score

    def export_detail_classification(self, 
                                     filename, 
                                     sku_data, 
                                     classified_data, 
                                     trend_column_prefix = ''):
        print("EXPORT CLASSIFICATION DETAILS ...")
        acc      = self.module.acceptance_threshold
        details  = pd.DataFrame(sku_data, copy=True)

        # remove summary column
        # del details['summary']
        # details.reset_index()

        for col in classified_data.columns:
            classified_data[col] = classified_data[col].apply(lambda x: 1 
                                                                    if x >= acc 
                                                                    else 0
                                                             )
        classified_data.columns = [trend_column_prefix + col for col in classified_data.columns.tolist()]       
        # output_data           = pd.concat([details, classified_data], axis=1)
        output_data             = pd.concat( [ pd.DataFrame({'sku': details.sku.values }), 
                                               classified_data
                                             ], 
                                             axis = 1)
        self.s3.write_csv(output_data, 
                          filename)
        print("\t{} rows written to {}".format( len(output_data), 
                                                filename)
              )
        return output_data

class ColorBaseClassifier(TrendClassifier):
    def __init__(self, 
                 configuration, 
                 module):
        print("T R E N D   V I S U A L    C L A S S I F I E R")
        self.config = configuration
        self.module = configuration.modules[module]
        self.trends = self.module.trends
        self.s3     = storage.create_storage(self.config)

    def classify(self, exp_name, model):
        input       = self.s3.read_csv(self.config.paths.unlabeled_data)
        for col in ['country', 
                    'days_since_activation_date', 
                    'image_url', 
                    'short_description', 
                    'summary', 
                    'brand', 
                    'sub_cat_type', 
                    'color_family', 
                    'product_type']:
            del input[col]
        for trend in self.trends:
            trend_colors = self.module.rules[trend]
            print('Classifying trend {}...'.format(trend), 
                   end = ''
                  )
            input[trend] = input.color.apply(lambda x: 1 
                                                    if (x in trend_colors) 
                                                    else 0)
        del input['color']
        print('\nClassifying trend Other...', 
               end = ''
              )
        input['Other'] = input[self.trends].apply(lambda r: 0 
                                                      if sum(r) > 0 
                                                      else 1, 
                                                  raw  = True, 
                                                  axis = 1)
        filename = self.config.paths.classification_output_data.format(exp_name)
        print('\nExport data to {}...'.format(filename))
        self.s3.write_csv(input, 
                          filename)


class BrandBaseClassifier(TrendClassifier):
    def __init__(self, 
                 configuration, 
                 module):
        print("BRAND-BASED TREND CLASSIFIER")
        self.config = configuration
        self.module = configuration.modules[module]
        self.trends = self.module.trends
        self.s3     = storage.create_storage(self.config)

    
    def classify(self, exp_name, model):
        input       = self.s3.read_csv(self.config.paths.unlabeled_data)
        for col in ['country', 
                    'days_since_activation_date', 
                    'image_url', 
                    'short_description', 
                    'summary', 
                    'color', 
                    'sub_cat_type', 
                    'color_family', 
                    'product_type']:
            del input[col]
        for trend in self.trends:
            trend_brands = self.module.rules[trend]
            print('Classifying trend {}...'.format(trend), 
                   end = ''
                  )
            input[trend] = input.brand.apply(lambda x: 1 
                                                    if (x in trend_brands) 
                                                    else 0)
        del input['brand']
        print('\nClassifying trend Other...', 
               end = ''
              )
        input['Other'] = input[self.trends].apply(lambda r: 0 
                                                      if sum(r) > 0 
                                                      else 1, 
                                                  raw  = True, 
                                                  axis = 1)
        filename = self.config.paths.classification_output_data.format(exp_name)
        print('\nExport data to {}...'.format(filename))
        self.s3.write_csv(input, 
                          filename)


                          
#-#-#-#-#----CLASS----#-#-#-#-#						  
class OccasionClassifier(TrendClassifier):
    def __init__(self, 
                 configuration, 
                 module):
        print("OCCASION TREND CLASSIFIER")
        self.config = configuration
        self.module = configuration.modules[module]
        self.s3     = storage.create_storage(self.config)
   
    def classify(self, exp_name, model):
        input       = self.s3.read_csv(self.config.paths.unlabeled_data)
        for col in ['brand', 
                    'country', 
                    'days_since_activation_date', 
                    'image_url', 
                    'short_description', 
                    'summary', 
                    'color', 
                    'sub_cat_type']:
            del input[col]
        input['Basic'] = 1
        input['Other'] = 0
        filename = self.config.paths.classification_output_data.format(exp_name)
        print('\nExport data to {}...'.format(filename))
        self.s3.write_csv(input, 
                          filename)

def create_classifier(config: ObjDict, module_name: str) -> TrendClassifier:
    module     = config.modules[module_name]
    classifier = module.classifier if 'classifier' in module.keys() else 'CNN'
    if classifier == "CNN":
        return CNNTrendClassifier(config, module_name)
    if classifier == 'brandbase':
        return BrandBaseClassifier(config, module_name)
    if classifier == 'colorbase':
        return ColorBaseClassifier(config, module_name)
    if classifier == "occasion":
        return OccasionClassifier(config, module_name)
    return None