## model_A is a placeholder model
#%%
import tensorflow as tf
from tensorflow import keras 
import mlflow
import sys
import os



try:
    gpu_no = sys.argv[1]
    '''Set visible GPU'''
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no
except:
    gpu_no = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_no


## record everything in mlflow
database_uri = 'http://127.0.0.1:5000'
run_name = "Model A"
mlflow.set_tracking_uri(database_uri)
mlflow.tensorflow.autolog()

## create a tag for the run
tags = {
    "Run#":f"{sys.argv[2]}",
}

with mlflow.start_run(run_name=run_name, tags=tags):
    with tf.device('/device:GPU:' + gpu_no):
        ## This is from the tensorflow documentation.
        ## https://www.tensorflow.org/guide/keras/training_with_built_in_methods
        #%%-------------------------------------------------
        inputs = keras.Input(shape=(784,), name="digits")
        x = keras.layers.Dense(64, activation="relu", name="dense_1")(inputs)
        x = keras.layers.Dense(64, activation="relu", name="dense_2")(x)
        outputs = keras.layers.Dense(10, activation="softmax", name="predictions")(x)

        model = keras.Model(inputs=inputs, outputs=outputs)


        #%%-------------------------------------------------
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Preprocess the data (these are NumPy arrays)
        x_train = x_train.reshape(60000, 784).astype("float32") / 255
        x_test = x_test.reshape(10000, 784).astype("float32") / 255

        y_train = y_train.astype("float32")
        y_test = y_test.astype("float32")

        # Reserve 10,000 samples for validation
        x_val = x_train[-10000:]
        y_val = y_train[-10000:]
        x_train = x_train[:-10000]
        y_train = y_train[:-10000]


        #%%-------------------------------------------------

        model.compile(
            optimizer=keras.optimizers.RMSprop(),  # Optimizer
            # Loss function to minimize
            loss=keras.losses.SparseCategoricalCrossentropy(),
            # List of metrics to monitor
            metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

        #%%-------------------------------------------------
        print("Training")
        history = model.fit(
            x_train,
            y_train,
            batch_size=64,
            epochs=10,
            validation_data=(x_val, y_val),
        )
