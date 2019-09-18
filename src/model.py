import tensorflow as tf

class Encoder(tf.keras.Model):
    '''
        Docstring:
            Creates the encoder model using GRU
    '''
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        '''
            Docstring:
                Creates a GRU model to encode input data

            Params:
                vocab_size: Int > 0. Size of the vocabulary, i.e. maximum integer index + 1
                embedding_dim: int >= 0. Dimension of the dense embedding
                enc_units: Int > 0. Dimensionality of the output space
                batch_sz: Int > 0. Training batch size
        '''
        super().__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.gru = tf.keras.layers.GRU(units=self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        '''
            Params:
                x: List. List of input data
                hidden: Vector. Hidden state of GRU. Initial state is a vector or zeroes
            
            Returns:
                output: Vector. Output of GRU. Shape = (Batch size x Sequence Length x Units in GRU)
                state: Vector. Hidden state of GRU
        '''
        x = self.embedding(x)
        output, state = self.gru(inputs=x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        '''
            Docstring:
                Initializes the GRU hidden state with zeroes, shape = (Batch size X Units in GRU)
            
            Returns:
                Initialized hidden state
        '''
        return tf.zeros(shape=(self.batch_sz, self.enc_units))

class Decoder(tf.keras.Model):
    '''
        Docstring:
            Creates the decoder model using GRU and Fully Connected layers for final predictions
    '''
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        '''
            Docstring:
                Creates a GRU model to decode input data

            Params:
                vocab_size: Int > 0. Size of the vocabulary, i.e. maximum integer index + 1
                embedding_dim: int >= 0. Dimension of the dense embedding
                dec_units: Int > 0. Dimensionality of the output space
                batch_sz: Int > 0. Training batch size
        '''
        super().__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units=self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        # self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        # context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        # x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)
        
        # return x, state, attention_weights
        return x, state