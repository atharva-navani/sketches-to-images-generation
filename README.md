# Sketches to Image Generation using Conditional Generative Adversarial Networks (cGAN)

This project aims to generate realistic images from sketches using Conditional Generative Adversarial Networks (cGAN). The cGAN architecture consists of a Generator and a Discriminator network that work together to produce high-quality images.


#Sketches to image generation using GANs

## OUTPUT

<img width="1126" alt="Screenshot 2024-03-23 at 12 16 45 PM" src="https://github.com/atharva-navani/sketches-to-images-generation/assets/164495332/89e76923-7ca7-4dc2-9874-30fbc71fd005">

![Original](https://github.com/atharva-navani/sketches-to-images-generation/assets/164495332/99bf500a-b7d2-4bc8-bc5a-00649b586771)


![Original-3](https://github.com/atharva-navani/sketches-to-images-generation/assets/164495332/3f6f27f5-4acb-4eb0-92fb-3a6735859aed)



### Steps:

**1.Data Collection**

http://mmlab.ie.cuhk.edu.hk/archive/facesketch.html

**2.Defining Paths**

**3.Data Preprocessing**

**4.Defining 3 key Models:**

  Generator

  Discriminator

  Combined

**5.Setting up losses and optimizers**

**6.Training the model**

**7.Testing the trained model**



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

## Limitations:

1. **Limited Dataset**: The performance of the model may be limited by the availability and diversity of the dataset used for training. A larger and more diverse dataset could potentially improve the quality of generated photos.

2. **GPU Computation Power**: The computational demands of training deep learning models, especially GANs, can be substantial. Limited GPU computational power may result in longer training times or may even restrict the size and complexity of the model that can be trained effectively.

3. **Hardware Incompetency**: Inadequate hardware infrastructure, including CPU, GPU, and memory resources, may constrain the scalability and efficiency of training the model.

4. **Artistic Interpretation**: While the model aims to generate realistic photos from sketches, it may lack the artistic intuition and context understanding that human artists possess. Thus, generated photos may not always perfectly reflect the intended interpretation of the sketches.

### Additional Considerations:

5. **Subjective Quality Evaluation**: Evaluating the quality of generated photos is inherently subjective and may vary depending on individual preferences and perspectives. Objective metrics may not always capture the nuanced aspects of image quality.

## Future Scope:

1. **Police Sketches for Recognizing Suspects**: The model could be extended for generating realistic images from police sketches, aiding law enforcement agencies in identifying and apprehending suspects more effectively.

2. **Tracking Cybercriminals**: By generating realistic images from digital sketches or descriptions of cybercriminals, the model could assist in tracking and apprehending individuals involved in cybercrimes, such as hacking or online fraud.

3. **Artistic Collaboration**: Beyond practical applications, the model could facilitate artistic collaboration by generating photorealistic images from conceptual sketches, enabling artists to visualize their ideas more vividly and efficiently.

4. **Fashion and Design Industry**: In the fashion and design industry, the model could be utilized to quickly prototype and visualize design concepts, accelerating the creative process and reducing the time-to-market for new products.

5. **Medical Imaging**: Extending the model to generate realistic medical images from diagnostic sketches or descriptions could aid medical professionals in visualizing patient conditions and treatment plans more intuitively.
## References
https://www.researchgate.net/publication/362481944_Sketch_to_Image_Using_Generative_Adversarial_Networks_GAN
https://www.researchgate.net/publication/268079622_Conditional_Generative_Adversarial_Nets
https://www.researchgate.net/publication/358617446_Sketch_to_image_generation_using_generative_adversarial_network

## License

This project is licensed under the MIT License 

## Acknowledgments

- This project was inspired by the original cGAN paper.
- Special thanks to the contributors of the open-source libraries used in this project.
