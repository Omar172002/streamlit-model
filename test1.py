from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential(
    [
        layers.Dense(units=1, input_shape=[10])
    ]
)

print(model.summary())

w, b = model.weights
print(w)
print(w.numpy())

print(b)
print(b.numpy())

