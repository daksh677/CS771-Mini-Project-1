import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Model


class MLModel():
    def __init__(self) -> None:
        pass
    
    def train(self, X, y):
        pass
    
    def predict(self, X):
        pass
    
class TextSeqModel(MLModel):
    def __init__(self) -> None:
        """
        Intializes the text sequence model
        """
        print(f"Intializing text sequence model...")
        train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
        train_seq_X = np.array(train_seq_df['input_str'].tolist())
        train_seq_Y = np.array(train_seq_df['label'].tolist())

        self.model = None
        self.fit(train_seq_X, train_seq_Y)

    def fit(self, X, y):
        """
        X: Text sequence data for training
        y: Label data for training

        Fitting the text sequence model
        """
        print(f"Fitting text sequence model...")
        X = np.array([[int(char) for char in seq] for seq in X])

        vocab_size = 10  
        sequence_length = 50  
        embedding_dim = 16 
        num_classes = 2 

        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length))
        self.model.add(Conv1D(filters=8, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))

        self.model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))

        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))  
        self.model.add(Dropout(0.5))  
        self.model.add(Dense(num_classes, activation='softmax'))  

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

    def predict(self, X):
        """ 
        X: Text sequence data for prediction
        Predicting the text sequence model
        """
        print(f"Predicting text sequence model...")
        X = np.array([[int(char) for char in seq] for seq in X])
        Y_prob = self.model.predict(X)
        Y_pred = np.argmax(Y_prob, axis=1)
        return Y_pred
    
    
class EmoticonModel(MLModel):
    def __init__(self) -> None:
        """ 
        Intializes the emoticon model
        """
        print(f"Intializing emoticon model...")
        train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
        train_emoticon_X = np.array(train_emoticon_df['input_emoticon'].tolist())
        train_emoticon_Y = np.array(train_emoticon_df['label'].tolist())

        self.model = None
        self.tokenizer = None
        self.fit(train_emoticon_X, train_emoticon_Y)

    def fit(self, X, y):
        """ 
        X: Emoticon data for training
        y: Label data for training
        Fitting the emoticon model using Nueral Network
        """
        print(f"Fitting emoticon model...")
        self.tokenizer = Tokenizer(char_level=True, oov_token='<OOV>') 
        self.tokenizer.fit_on_texts(X)
        X = np.array(self.tokenizer.texts_to_sequences(X))

        input_length = 13
        embedding_dim = 16
        dense_units = 16
        vocab_size = len(self.tokenizer.word_index) + 1

        self.model = Sequential()
        self.model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
        self.model.add(Flatten())
        self.model.add(Dense(dense_units, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    def predict(self, X):
        """
        X: Emoticon data for prediction
        Predicting the emoticon model
        """
        print(f"Predicting emoticon model...")
        X = np.array(self.tokenizer.texts_to_sequences(X))
        Y_prob = self.model.predict(X)
        Y_pred = (Y_prob > 0.5).astype(int).flatten()
        return Y_pred

    
class FeatureModel(MLModel):
    def __init__(self) -> None:
        """
        Feature model
        """
        print("Initializing feature model...")
        train_feat_df = np.load("datasets/train/train_feature.npz", allow_pickle=True)
        train_feat_X = train_feat_df['features']
        train_feat_Y = train_feat_df['label']

        self.model = None
        self.pca = None
        self.fit(train_feat_X, train_feat_Y)

    def fit(self, X, y):
        """
        X: Feature data for training
        y: Label data for training

        Fitting the feature model
        """
        print("Fitting feature model...")
        self.pca = PCA(n_components=200)
        X_flatten = X.reshape(X.shape[0], -1)
        X_pca = self.pca.fit_transform(X_flatten)
        self.model = SVC(kernel='rbf')
        self.model.fit(X_pca, y)

    def predict(self, X):
        """
        X: Feature data for prediction
        Predicting the feature model
        """
        print("Predicting feature model...")
        X_flatten = X.reshape(X.shape[0], -1)
        X_pca = self.pca.transform(X_flatten)
        return self.model.predict(X_pca)
    
class CombinedModel(MLModel):
    def __init__(self) -> None:
        """
        Intializes the combined model
        """
        print("Initializing combined model...")
        train_feat_df = np.load("datasets/train/train_feature.npz", allow_pickle=True)
        train_feat_X = train_feat_df['features']

        train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
        train_emoticon_X = np.array(train_emoticon_df['input_emoticon'].tolist())

        train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
        train_seq_X = np.array(train_seq_df['input_str'].tolist())
        train_seq_Y = np.array(train_seq_df['label'].tolist())

        self.X1_scaler = None
        self.X1_pca = None

        self.X2_scaler = None
        self.X2_tokenizer = None

        self.X3_scaler = None

        self.model = None
        self.fit(train_feat_X, train_emoticon_X, train_seq_X, train_seq_Y)

    def fit(self, X1, X2, X3, y):
        """
        X1: Feature data for training
        X2: Emoticon data for training
        X3: Text data for training
        y: Label data for training

        Fitting the combined model
        """
        print("Fitting combined model...")
        # Feature
        self.X1_scaler = StandardScaler()
        X1_flatten = X1.reshape(X1.shape[0], -1)
        X1_scaled = self.X1_scaler.fit_transform(X1_flatten)

        self.X1_pca = PCA(n_components=150)
        X1_pca = self.X1_pca.fit_transform(X1_scaled)

        # Emoticon
        self.X2_scaler = StandardScaler()
        self.X2_tokenizer = Tokenizer(char_level=True, oov_token='<OOV>')
        self.X2_tokenizer.fit_on_texts(X2)
        X2_token = np.array(self.X2_tokenizer.texts_to_sequences(X2))
        X2_token_scaled = self.X2_scaler.fit_transform(X2_token)

        # Text Sequence
        X3 = np.array([[int(char) for char in seq] for seq in X3])
        self.X3_scaler = StandardScaler()
        X3_scaled = self.X3_scaler.fit_transform(X3)

        X = np.concatenate((X1_pca, X2_token_scaled, X3_scaled), axis=1)

        self.model = SVC(kernel='rbf')
        self.model.fit(X, y)
        
    def predict(self, X1, X2, X3):
        """
        X1: Feature data for prediction
        X2: Emoticon data for prediction
        X3: Text data for prediction

        Predicting the combined model
        """
        print("Predicting combined model...")
        X1_flatten = X1.reshape(X1.shape[0], -1)
        X1_scaled = self.X1_scaler.transform(X1_flatten)
        X1_pca = self.X1_pca.transform(X1_scaled)

        X2_token = np.array(self.X2_tokenizer.texts_to_sequences(X2))
        X2_token_scaled = self.X2_scaler.transform(X2_token)

        X3 = np.array([[int(char) for char in seq] for seq in X3])
        X3_scaled = self.X3_scaler.transform(X3)

        X = np.concatenate((X1_pca, X2_token_scaled, X3_scaled), axis=1)
        return self.model.predict(X)
    
    
def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

if __name__ == '__main__':
    # read datasets
    test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)['features']
    test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()
    test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str'].tolist()
    
    # your trained models 
    feature_model = FeatureModel()
    text_model = TextSeqModel()
    emoticon_model  = EmoticonModel()
    best_model = CombinedModel()
    
    # predictions from your trained models
    pred_feat = feature_model.predict(test_feat_X)
    pred_emoticons = emoticon_model.predict(test_emoticon_X)
    pred_text = text_model.predict(test_seq_X)
    pred_combined = best_model.predict(test_feat_X, test_emoticon_X, test_seq_X)
    
    # saving prediction to text files
    save_predictions_to_file(pred_feat, "pred_feat.txt")
    save_predictions_to_file(pred_emoticons, "pred_emoticon.txt")
    save_predictions_to_file(pred_text, "pred_text.txt")
    save_predictions_to_file(pred_combined, "pred_combined.txt")