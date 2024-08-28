#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam


# In[2]:


# Load data
data = pd.read_csv('final_comp_data.csv')

# Preprocess data
compositions = data.iloc[:, :13].values
yield_strength = data.iloc[:, 13].values

compositions_normalized = compositions

scaler_yield = MinMaxScaler()
yield_normalized = scaler_yield.fit_transform(yield_strength.reshape(-1, 1))


# In[3]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Input, Flatten, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

# Generator model
def build_generator(latent_dim, output_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(output_dim, activation='tanh'))
    return model

# Discriminator model
def build_discriminator(input_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Encoder model
def build_encoder(input_dim, latent_dim):
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(latent_dim))
    return model

# Compile GAN
def build_gan(generator, discriminator):
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.28), metrics=['accuracy'])
    z = Input(shape=(latent_dim,))
    alloy = generator(z)
    discriminator.trainable = False
    validity = discriminator(alloy)
    combined = Model(z, validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.28))
    return combined


# In[4]:


import matplotlib.pyplot as plt

latent_dim = 2
output_dim = 13
epochs = 1000
batch_size = 32
sample_interval = 64

generator = build_generator(latent_dim, output_dim)
discriminator = build_discriminator(output_dim)
gan = build_gan(generator, discriminator)
encoder = build_encoder(output_dim, latent_dim)

# Train the GAN
half_batch = int(batch_size / 2)

d_losses = []
g_losses = []

for epoch in range(epochs):
    idx = np.random.randint(0, compositions_normalized.shape[0], half_batch)
    real_compositions = compositions_normalized[idx]
    noise = np.random.normal(0, 1, (half_batch, latent_dim))
    fake_compositions = generator.predict(noise)
    
    d_loss_real = discriminator.train_on_batch(real_compositions, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_compositions, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_y = np.array([1] * batch_size)
    g_loss = gan.train_on_batch(noise, valid_y)
    
    d_losses.append(d_loss[0])
    g_losses.append(g_loss)
    
    if epoch % sample_interval == 0:
        print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}%] [G loss: {g_loss}]")

# Plot training losses
plt.figure(figsize=(10, 7))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[5]:


# Autoencoder for reconstruction
real_compositions_input = Input(shape=(output_dim,))
encoded_compositions = encoder(real_compositions_input)
reconstructed_compositions = generator(encoded_compositions)

autoencoder = Model(real_compositions_input, reconstructed_compositions)
autoencoder.compile(loss='mse', optimizer=Adam(0.0002, 0.5))

# Train the autoencoder and capture loss values
autoencoder_history = autoencoder.fit(compositions_normalized, compositions_normalized, epochs=1000, batch_size=batch_size, verbose=1)

# Project known compositions into the latent space
latent_vectors = encoder.predict(compositions_normalized)
print("Latent vectors for known compositions:")
print(latent_vectors)

# Plot the autoencoder training loss
plt.figure(figsize=(10, 5))
plt.plot(autoencoder_history.history['loss'], label='Autoencoder Loss')
plt.title('Autoencoder Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('los_epoch_VAE.jpg', dpi=600)
plt.show()

# Plot the latent space and save the figure
plt.figure(figsize=(10, 7))
plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=yield_strength, marker='v', cmap='viridis', s=100)
plt.colorbar(label='Yield Strength')
plt.title('2D Latent Space')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.savefig('latent_space_plot.jpg', dpi=600)
plt.show()


# In[6]:


# Generate 10,000 new compositions
num_samples = 1000
noise = np.random.normal(0, 1, (num_samples, latent_dim))
new_compositions = generator.predict(noise)

# Denormalize the new compositions
new_compositions_denormalized = scaler_compositions.inverse_transform(new_compositions)

# Export to Excel
new_compositions_df = pd.DataFrame(new_compositions_denormalized, columns=[f'Element_{i+1}' for i in range(output_dim)])
new_compositions_df.to_excel('generated_compositions.xlsx', index=False)

print("Generated compositions exported to 'generated_compositions.xlsx'")

# Project the generated compositions into the latent space
generated_latent_vectors = encoder.predict(new_compositions)


# In[ ]:


# Plot the latent space and save the figure
plt.figure(figsize=(10, 7))
plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=yield_strength, cmap='viridis', s=0, alpha=0.5)
plt.scatter(generated_latent_vectors[:, 0], generated_latent_vectors[:, 1], c='orchid', s=100, alpha=0.7)
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.legend()
plt.savefig('latent_space_plot_with_generated_CoCrFeMnNiAlCu.jpg', dpi=600)
plt.show()


# In[ ]:





# In[ ]:




