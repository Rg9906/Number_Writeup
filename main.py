from ast import Load

# import tensorflow as tf

# # Load MNIST dataset
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # Normalize pixel values (0-255 -> 0-1)
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# # Build model
# model = tf.keras.models.Sequential([
#     tf.keras.Input(shape=(28, 28)),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(128, activation="relu"),
#     tf.keras.layers.Dense(10, activation="softmax")
# ])

# # Compile model
# model.compile(
#     optimizer="adam",
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# # Train model
# model.fit(x_train, y_train, epochs=25wait mnist)

# # Evaluate model
# loss, accuracy = model.evaluate(x_test, y_test)

# print("\nTest Loss:", loss)
# print("Test Accuracy:", accuracy)

# # Save model
# model.save("epic_num_reader.keras")

# print("\nModel loaded successfully from epic_num_reader.keras")

# loss,accuracy=model.evaluate(x_test,y_test)
# print("\nTest Loss:", loss)
# print("Test Accuracy:", accuracy)
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model("epic_num_reader.keras")

print("Model loaded successfully!\n")

for i in range(1, 10):

    filename = f"Digit{i}.png"
    path = os.path.join("digits", filename)

    if not os.path.isfile(path):
        print(f"{filename} not found")
        continue

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (28, 28))

    # Only invert if your digit is black on white
    img = np.invert(img)

    img = img / 255.0

    img_input = np.array([img])

    prediction = model.predict(img_input, verbose=0)

    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    print(
        f"Actual: {i} | Predicted: {predicted_digit} | Confidence: {confidence:.2f}%"
    )

    plt.imshow(img, cmap="binary")
    plt.title(f"Actual: {i}  Predicted: {predicted_digit}")
    plt.axis("off")
    plt.show()
