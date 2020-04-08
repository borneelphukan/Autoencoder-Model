import matplotlib
matplotlib.use("Agg")
from AutoEncoder import Autoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--samples", type = int, default = 8)
ap.add_argument("-o", "--output", type = str, default = "output.png")
ap.add_argument("-p", "--plot", type = str, default = "plot.png")
args = vars(ap.parse_args())

EPOCHS = 25
BATCH_SIZE = 32

print("STEP 1: LOADING DATASET...")
((train_x, _), (test_x, _)) = mnist.load_data()

train_x = np.expand_dims(train_x, axis = -1)
test_x = np.expand_dims(test_x, axis = -1)
train_x = train_x.astype("float32")/255.0
test_x = test_x.astype("float32")/255.0

print("STEP 2: BUILDING AUTOENCODER...")
(encoder, decoder, autoencoder) = Autoencoder.build(28, 28, 1)
custom_optimizer = Adam(lr = 0.001)
autoencoder.compile(loss = "mean_squared_error", optimizer = custom_optimizer)

model = autoencoder.fit(train_x, train_x, validation_data=(test_x, test_x), epochs = EPOCHS, batch_size = BATCH_SIZE)

loss = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(loss, model.history["loss"], label="train_loss")
plt.plot(loss, model.history["val_loss"], label="val_loss")
plt.title("Training Loss v/s Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["loss_curve"])

print("STEP 3: PREDICTION...")
decoded = autoencoder.predict(test_x)
outputs = None

for i in range(0, args["samples"]):
	original = (test_x[i] * 255).astype("uint8")
	recon = (decoded[i] * 255).astype("uint8")
	output = np.hstack([original, recon])
	if outputs is None:
		outputs = output
	else:
		outputs = np.vstack([outputs, output])
cv2.imwrite(args["recreated_data"], outputs)

print("Loss Curve has been saved as loss_curve.png")
print("Recreated output has been saved as recreated_data.png")