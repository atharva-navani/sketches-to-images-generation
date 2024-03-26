# Sketches to Image Generation using Conditional Generative Adversarial Networks (cGAN)

This project aims to generate realistic images from sketches using Conditional Generative Adversarial Networks (cGAN). The cGAN architecture consists of a Generator and a Discriminator network that work together to produce high-quality images.

## Generator

### Input Layer
- Input images: (128, 128, 3) representing sketches with three color channels (RGB).

### Bottom Layer Sample
- Begins with three Convolution layers:
  - `conv1_layer`: Applies 32 filters of size 7x7 with stride (1, 1), followed by BatchNormalization and ReLU activation.
  - `conv2_layer`: Applies 64 filters of size 3x3 with stride (2, 2), followed by BatchNormalization and ReLU activation.
  - `conv3_layer`: Applies 128 filters of size 3x3 with stride (2, 2), followed by BatchNormalization and ReLU activation.

### Remaining Blocks
- Contains six remaining blocks, each consisting of two Convolution layers followed by BatchNormalization and ReLU activation. The output of the first convolution layer is added to the output of the second convolution layer within each block.

### Sample Layer
- Includes two Transpose Convolution layers (`transpose1` and `transpose2`) to upsample the feature map to the original input resolution.

### Output Layer
- Applies a convolution operation with three filters of size 3x3 to generate the final image.

## Discriminator

### Input Level
- Expects input images of size (128, 128, 3).

### Revolutionary Levels
- Consists of several Convolutional layers with LeakyReLU activation functions to extract features from input images.

### Activation Functions
- LeakyReLU activation functions are applied after each Convolutional layer to introduce non-linearity.

### Batch Normalization
- Applied after several Convolutional layers for stabilizing the training process.

### Flat Level
- Output from Convolutional layers is flattened into a 1-dimensional tensor.

### Output Level
- Passed through a dense layer with a neuron and a sigmoid activation function to output the probability that the input image is real.

## Training Strategies

- **Learning Rate Decay**: Learning rate decay is applied to adjust the learning rate as training progresses, which can help convergence.
- **Adversarial Training**: Discriminator and generator are alternately trained to compete against each other, aiming for improved performance.

## Training Loop

The code iterates over epochs, adjusting the learning rate based on epoch number.

Within each epoch, it iterates over batches of data.

For each batch:

- It selects a batch of sketches (X) and their corresponding real photos (Y).
- The generator model generates fake photos (fake_photos) from the sketches.
- The fake photos are resized to match the size of real photos.
- Labels (discriminator_Y) are created for the discriminator, where real photos are labeled as 1 and fake photos as 0.
- Discriminator inputs (discriminator_X) are created by concatenating real and fake photos.
- The discriminator inputs are rescaled to the range [-1, 1].
- The discriminator is trained on this batch using `train_on_batch`.
- The generator is then trained using the sketches and the expected real photos, and it aims to fool the discriminator by generating photos that the discriminator will classify as real.
- Epoch and losses (generator loss and discriminator loss) are printed.

## Model Training

- The generator and discriminator models are trained alternately in each batch.
- The discriminator is trained to distinguish between real and fake images.
- The generator is trained to generate images that are indistinguishable from real images by the discriminator.

## Output

For each epoch, the batch number and corresponding generator loss and discriminator loss are printed.

Overall, this code represents a basic GAN training loop for image generation tasks, with specific emphasis on generating realistic photos from sketches.

## Dependencies

- `numpy`: NumPy is a library for numerical computations in Python.
- `cv2`: OpenCV is a library mainly aimed at real-time computer vision.

## Getting Started

1. Clone this repository.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Train the cGAN model using the provided dataset.
4. Generate images from sketches using the trained model.

## Contributing

Contributions are welcome! Please feel free to submit a pull request with any improvements or additional features.

## License

This project is licensed under the MIT License 

## Acknowledgments

- This project was inspired by the original cGAN paper.
- Special thanks to the contributors of the open-source libraries used in this project.
