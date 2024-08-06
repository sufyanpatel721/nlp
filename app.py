from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import custom_object_scope
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, Add, Embedding, GlobalAveragePooling1D, Input

# Custom TransformerBlock class
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "d_model": self.mha.key_dim,
            "num_heads": self.mha.num_heads,
            "dff": self.ffn.layers[0].units,
            "rate": self.dropout1.rate,
        })
        return config

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Set maxlen to the same value used during training
maxlen = 46

app = Flask(__name__)

# Load the model with custom object scope
with custom_object_scope({'TransformerBlock': TransformerBlock}):
    model = load_model('sentiment_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    tweet = data['tweet']
    
    # Preprocess the tweet
    tweet_seq = tokenizer.texts_to_sequences([tweet])
    tweet_pad = pad_sequences(tweet_seq, maxlen=maxlen, truncating='post', padding='post')
    
    # Predict sentiment
    prediction = model.predict(tweet_pad)
    sentiment = np.argmax(prediction, axis=-1)[0]
    
    return jsonify({'sentiment': int(sentiment)})

if __name__ == '__main__':
    app.run(debug=True)
