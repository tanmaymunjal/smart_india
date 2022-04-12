# make all necessary imports

import tensorflow as tf
from tensorflow import keras
from keras import layers


# program to define basic layer classes that shall be imported and used later

# <------------sound processing part--------------------->

# function to encode info coming to neural network

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlength, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlength, output_dim=embed_dim)
        self.maxlength=maxlength

    def call(self,x):
        maxlength = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlength, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


# function to define mask to prevent future info leak to AI algorithm
def mask(batch, size):
    x, y = tf.expand_dims(tf.range(size), 1), tf.range(size)
    mask2 = x >= y
    mask2 = tf.reshape(mask2, (1, size, size))
    mask2 = tf.tile(mask2, (batch, 1, 1))

    return mask2


# program to define transformer blocks
class TransformersBlock(layers.Layer):
    def __init__(self,embed_dim, num_heads, dropout=0):
        super(TransformersBlock, self).__init__()

        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dropout1 = layers.Dropout(dropout)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)

        self.feed_forward_network = keras.Sequential([
            layers.Dense(embed_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])

        self.droupout2 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, batch_size, seq_len, masked=0):
            if masked == 1:
                what_out = self.attention(inputs, inputs, attention_mask=mask(batch_size, seq_len))
            else:
                what_out = self.attention(inputs, inputs)
            droupout1_out = self.dropout1(what_out)
            norm1_out = self.norm1(droupout1_out)
            combined_out = norm1_out + inputs
            ffn_out = self.feed_forward_network(combined_out)
            droupout2_out = self.droupout2(ffn_out)
            norm2_out = self.norm2(droupout2_out)
            combined2_out = norm2_out + combined_out

            return combined2_out



# -----------------image,radar, and lidar processing parts -------------------

class ChannelAttention(layers.Layer):
    def __init__(self, filters, ratio):
        super(ChannelAttention, self).__init__()
        self.filters = filters
        self.ratio = ratio

    def build(self):
            self.shared_layer_one = layers.Dense(self.filters // self.ratio, activation='relu',
                                                 kernel_initializer='he_normal', use_bias=True,
                                                 bias_initializer='zeros')
            self.shared_layer_two = layers.Dense(self.filters, kernel_initializer='he_normal', use_bias=True,
                                                 bias_initializer='zeros')

    def call(self,inputs):
            # AvgPool
            avg_pool = layers.GlobalAveragePooling2D()(inputs)

            avg_pool = self.shared_layer_one(avg_pool)
            avg_pool = self.shared_layer_two(avg_pool)

            # MaxPool
            max_pool = layers.GlobalMaxPooling2D()(inputs)
            max_pool = layers.Reshape((1, 1, self.filters))(max_pool)

            max_pool = self.shared_layer_one(max_pool)
            max_pool = self.shared_layer_two(max_pool)

            attention = layers.Add()([avg_pool, max_pool])
            attention = layers.Activation('sigmoid')(attention)

            return layers.Multiply()([inputs, attention])


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

    def build(self):
            self.conv2d = layers.Conv2D(filters=1,
                                        kernel_size=self.kernel_size,
                                        strides=1,
                                        padding='same',
                                        activation='sigmoid',
                                        kernel_initializer='he_normal',
                                        use_bias=False)

    def call(self,inputs):
            # AvgPool
            avg_pool = layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(inputs)

            # MaxPool
            max_pool = layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(inputs)

            attention = layers.Concatenate(axis=3)([avg_pool, max_pool])

            attention = self.conv2d(attention)

            return layers.multiply([inputs, attention])


