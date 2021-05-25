import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

(train_data, test_data), info = tfds.load('imdb_reviews/subwords8k',
                                    split=(tfds.Split.TRAIN, tfds.Split.TEST),
                                    with_info=True, as_supervised=True)


encoder = info.features['text'].encoder

padded_shapes = ([None], ())
train_batches = train_data.shuffle(1000).padded_batch(10,
                                            padded_shapes=padded_shapes)
        
test_batches = test_data.shuffle(1000).padded_batch(10,
                                            padded_shapes=padded_shapes)

embedding_dim = 16
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
            optimizer='Adam',
            metrics=['accuracy'])

history = model.fit(train_batches, epochs=10, validation_data=(test_batches),
                    validation_steps=20)
