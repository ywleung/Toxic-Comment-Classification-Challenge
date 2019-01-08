import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import keras.backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.initializers import glorot_uniform

def read_glove_vecs(glove_file):
    """
    function to read GloVe word embedding
    """
    with open(glove_file, 'r', encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        
        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

def kfold_train(kfold, X_df, y_df, model, save_model_name):
    """
    function to train model with k fold
    """
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_df)):
        print('***************************************************')
        print('*******************   fold  %d   *******************' % fold)
        print('***************************************************')
        
        final_save_model_name = 'model/' + save_model_name + '_fold_%d'%fold + '.model'
        print(final_save_model_name)
        
        # split dataset into train set and validation set
        X_train = X_df[train_idx]
        y_train = y_df[train_idx]
        X_val = X_df[val_idx]
        y_val = y_df[val_idx]
        
        # display model summary in first training
        if fold==0:
            model.summary()
        
        # callbacks
        early_stopping = EarlyStopping(patience=5, verbose=1)
        model_checkpoint = ModelCheckpoint(final_save_model_name, save_best_only=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)
        
        # train model
        history = model.fit(X_train, y_train, batch_size=512, 
                            epochs=30, validation_data=[X_val, y_val],
                            callbacks=[early_stopping, model_checkpoint, reduce_lr])
        
        # reset model weights for next fold
        k_eval = lambda placeholder: placeholder.eval(session=K.get_session())
        initial_weights = model.get_weights()
        new_weights = [k_eval(glorot_uniform()(w.shape)) for w in initial_weights]
        model.set_weights(new_weights)
        # reset learning weights
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # display accuracy and loss in the end
        fig, (ax_score, ax_loss) = plt.subplots(1, 2, figsize=(15, 5))
        ax_score.plot(history.epoch, history.history['acc'], label='Train')
        ax_score.plot(history.epoch, history.history['val_acc'], label='Val')
        ax_score.set_title('Acc (Fold %d)'%fold)
        ax_score.legend()
        ax_loss.plot(history.epoch, history.history['loss'], label='Train')
        ax_loss.plot(history.epoch, history.history['val_loss'], label='Val')
        ax_loss.set_title('Loss (Fold %d)'%fold)
        ax_loss.legend()
        
def model_predict(model, save_model_name, num_fold, X_test, list_class):
    """
    function to create prediction csv
    """
    sample_submission = pd.read_csv('sample_submission.csv')
    sample_submission[list_class] = 0
    
    for i in range(num_fold):
        print('Processing model {} ...'.format(i))
        model_name = 'model/' + save_model_name + '_fold_%d'%i + '.model'
        model.load_weights(model_name)        
        pred = model.predict(X_test, batch_size=512, verbose=2)
        sample_submission[list_class] = sample_submission[list_class] + pred
        
    sample_submission[list_class] = sample_submission[list_class] / num_fold    
    csv_name = save_model_name + '_prediction.csv'
    sample_submission.to_csv(csv_name, index=False)