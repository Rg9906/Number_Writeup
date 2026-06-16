import tensorflow as tf

# Load MNIST dataset
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
# model.fit(x_train, y_train, epochs=3)

# # Evaluate model
# loss, accuracy = model.evaluate(x_test, y_test)

# print("\nTest Loss:", loss)
# print("Test Accuracy:", accuracy)

# Save model
#model.save("epic_num_reader.keras")

# print("\nModel loaded successfully from epic_num_reader.keras")

# loss,accuracy=model.evaluate(x_test,y_test)
# print("\nTest Loss:", loss)
# print("Test Accuracy:", accuracy)
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load saved model
model = tf.keras.models.load_model("epic_num_reader.keras")

print("Model loaded successfully!\n")

# Check current directory
print("Current Working Directory:", os.getcwd())
print()

# Process all PNG files in the digits folder
for filename in os.listdir("digits"):
    if filename.lower().endswith(".png"):

        path = os.path.join("digits", filename)

        try:
            # Read image in grayscale
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"Could not read {filename}")
                continue

            # Resize to MNIST size
            img = cv2.resize(img, (28, 28))

            # Invert colors (MNIST uses white digit on black background)
            img = np.invert(img)

            # Normalize
            img = img / 255.0

            # Add batch dimension
            img_input = np.array([img])

            # Predict
            prediction = model.predict(img_input, verbose=0)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            print(
                f"{filename} -> Predicted: {predicted_digit} "
                f"({confidence:.2f}% confidence)"
            )

            # Show image
            plt.imshow(img, cmap=plt.cm.binary)
            plt.title(
                f"{filename}\nPrediction: {predicted_digit} "
                f"({confidence:.2f}%)"
            )
            plt.axis("off")
            plt.show()

        except Exception as e:
            print(f"Error processing {filename}: {e}")
