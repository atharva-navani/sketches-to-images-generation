# sketches-to-images-generation

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

## Combined Model

### Input Layer
- Input images of size (64, 64, 3) fed into the Generator.

### Generator
- Produces generated images from input sketches.

### Resampling
- Generated images are upsampled using UpSampling2D to match the dimensions expected by the Discriminator.

### Freeze Discriminator Weights
- Discriminator weights are frozen to only update the Generator weights during training.

### Discriminator
- Predicts whether generated images are real or fake.

### Outputs
- Generated images from the Generator.
- Predictions from the Discriminator about the generated images.

## Getting Started

1. Clone this repository.
2. Install the necessary dependencies using `pip install -r requirements.txt`.
3. Train the cGAN model using the provided dataset.
4. Generate images from sketches using the trained model.

## Contributing

Contributions are welcome! Please feel free to submit a pull request with any improvements or additional features.

## License

This project is licensed under the MIT License.

## Acknowledgments

- This project was inspired by the original cGAN paper.
- Special thanks to the contributors of the open-source libraries used in this project.
