import tensorflow as tf
import numpy as np

def generate_confusion_matrix( data, labels ):
    mat = [ [ 0 for i in range(10) ] for j in range(10) ]
    
    predictions = np.argmax( model.predict( data ), axis = 1 )
    
    for i in range( data.shape[0] ):
        mat[ labels[i] ][ predictions[i] ] += 1
    
    for i in range(10):
        print( "\t".join( [ str(c) for c in mat[i] ] ) )


( x_train, y_train ), ( x_test, y_test ) = tf.keras.datasets.mnist.load_data()

train_in = np.reshape( x_train, (-1, 28, 28) ) / 255
test_in = np.reshape( x_test, (-1, 28, 28) ) / 255
train_out = tf.keras.utils.to_categorical( y_train, 10 )
test_out = tf.keras.utils.to_categorical( y_test, 10 )


digit_input = tf.keras.layers.Input( shape = (28,28) )
flatten_image = tf.keras.layers.Flatten()( digit_input )
dense_1 = tf.keras.layers.Dense( units = 100, activation = tf.nn.relu )( flatten_image )
dense_2 = tf.keras.layers.Dense( units = 50, activation = tf.nn.relu )( dense_1 )
logits = tf.keras.layers.Dense( units = 10, activation = None )( dense_2 )
probabilities = tf.keras.layers.Softmax()( logits )


model = tf.keras.Model( inputs = digit_input, outputs = probabilities )
model.compile( optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )

generate_confusion_matrix( test_in, y_test )

history = model.fit( train_in, train_out, epochs = 20 )

generate_confusion_matrix( test_in, y_test )