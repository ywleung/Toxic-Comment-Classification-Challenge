import keras.backend as K
from keras.layers import Input, Embedding, Bidirectional, CuDNNLSTM, GlobalMaxPool1D, Dense, Dropout
from keras.models import Model
from attention_layer import *

def biLSTM_model(embed_size, max_feature, max_len, embedding_matrix):
    """
    function to build a simple bi-directional LSTM model
    """
    K.clear_session()
    
    X_input = Input((max_len, ))
    X = Embedding(max_feature, embed_size, weights=[embedding_matrix])(X_input)
    X = Bidirectional(CuDNNLSTM(50, return_sequences=True))(X)
    X = GlobalMaxPool1D()(X)
    X = Dense(50, activation='relu')(X)
    X = Dropout(0.1)(X)
    X = Dense(6, activation='sigmoid')(X)
    
    model = Model(inputs=X_input, output=X)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def biLSTM_attention_model(embed_size, max_feature, max_len, embedding_matrix):
    """
    function to build a bi-directional LSTM with Attention layer model
    """
    K.clear_session()
    
    X_input = Input((max_len, ))
    X = Embedding(max_feature, embed_size, weights=[embedding_matrix])(X_input)
    X = Bidirectional(CuDNNLSTM(64, return_sequences=True))(X)
    X = Bidirectional(CuDNNLSTM(64, return_sequences=True))(X)
    X = Attention(max_len)(X)
    X = Dense(64, activation='relu')(X)
    X = Dense(6, activation='sigmoid')(X)
    
    model = Model(inputs=X_input, output=X)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model