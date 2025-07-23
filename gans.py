# Import required libraries
import tensorflow as tf                # Core deep learning library
from tensorflow.keras import layers    # Neural network layer components
import numpy as np                     # Numerical computing library
import os                             # Operating system interface
import matplotlib.pyplot as plt        # Plotting library

# Define hyperparameters for the GAN
BATCH_SIZE = 128     # Number of images to process in each training batch
BUFFER_SIZE = 60000  # Size of the shuffle buffer for dataset
LATENT_DIM = 100    # Dimension of the random noise input vector
EPOCHS = 10         # Number of training epochs (reduced from 50)

# Function to load the Fashion-MNIST dataset from saved files
def load_saved_data():
    # Load training images from a numpy file
    train_images = np.load("/content/drive/MyDrive/fashion_images/train/train_images.npy")
    
    # Normalize pixel values to range [-1, 1] for better training stability
    train_images = train_images.astype('float32') / 127.5 - 1
    
    # Add channel dimension for grayscale images (28x28x1)
    train_images = np.expand_dims(train_images, axis=-1)
    return train_images

# Function to prepare the dataset for training
def prepare_dataset(train_images):
    # Convert numpy array to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(train_images)
    
    # Shuffle and batch the dataset
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return dataset

# Function to visualize sample images from the dataset
def visualize_data(train_images):
    plt.figure(figsize=(10, 5))
    # Display 10 sample images
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(train_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()

# Function to build the Generator model
def build_generator():
    model = tf.keras.Sequential([
        # Input layer: Dense layer to process random noise
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(LATENT_DIM,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Reshape to start the deconvolution process
        layers.Reshape((7, 7, 256)),

        # First deconvolution layer
        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Second deconvolution layer
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        # Output layer: Generate 28x28x1 image
        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Function to build the Discriminator model
def build_discriminator():
    model = tf.keras.Sequential([
        # First convolutional layer
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        # Second convolutional layer
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        # Flatten and output layer
        layers.Flatten(),
        layers.Dense(1)  # Single output for binary classification
    ])
    return model

# Define loss function for GAN training
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Discriminator loss function
def discriminator_loss(real_output, fake_output):
    # Loss for real images (should be classified as 1)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # Loss for fake images (should be classified as 0)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Generator loss function
def generator_loss(fake_output):
    # Generator tries to make discriminator classify fake images as real (1)
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Define optimizers for both networks
generator_optimizer = tf.keras.optimizers.Adam(1e-4)      # Learning rate = 0.0001
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)  # Learning rate = 0.0001

# Load and prepare the dataset
train_images = load_saved_data()
dataset = prepare_dataset(train_images)

# Show sample images from the dataset
visualize_data(train_images)

# Create the generator and discriminator models
generator = build_generator()
discriminator = build_discriminator()

# Function to display intermediate generated images during training
def display_intermediate_images(generator, step):
    # Generate random noise
    noise = tf.random.normal([1, LATENT_DIM])
    # Generate image from noise
    generated_image = generator(noise, training=False)

    # Display the generated image
    plt.imshow(generated_image[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.title(f"Generated Image at Step {step}")
    plt.axis('off')
    plt.show()

# Single training step function
def train_step(images, step):
    # Generate random noise for fake images
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])

    # Use gradient tape to track gradients for both networks
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images
        generated_images = generator(noise, training=True)

        # Get discriminator outputs for real and fake images
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Calculate losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    # Calculate gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply gradients to update weights
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    # Display progress every 100 steps
    if step % 100 == 0:
        display_intermediate_images(generator, step)

    return gen_loss, disc_loss

# Main training loop function
def train(dataset, epochs):
    step = 0
    for epoch in range(epochs):
        gen_loss_avg = 0
        disc_loss_avg = 0
        # Train on batches
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, step)
            gen_loss_avg += gen_loss
            disc_loss_avg += disc_loss
            step += 1

        # Calculate average losses for the epoch
        gen_loss_avg /= len(dataset)
        disc_loss_avg /= len(dataset)

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs} | Generator Loss: {gen_loss_avg:.4f} | Discriminator Loss: {disc_loss_avg:.4f}")

        # Generate and display sample images after each epoch
        display_generated_images(generator, epoch + 1)

# Function to display multiple generated images
def display_generated_images(generator, epoch, examples=16):
    # Generate random noise
    noise = tf.random.normal([examples, LATENT_DIM])
    # Generate images from noise
    generated_images = generator(noise, training=False)

    # Display grid of generated images
    plt.figure(figsize=(10, 10))
    for i in range(examples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.suptitle(f"Epoch {epoch} Generated Images", fontsize=16)
    plt.show()

# Start the training process
train(dataset, EPOCHS)