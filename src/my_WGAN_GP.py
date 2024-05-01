import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import grad
import numpy as np
import pandas as pd
from tqdm import tqdm

# Create directory for saving models if it doesn't exist
os.makedirs(r"C:\Users\mokht\Desktop\GAN-Drug-Generator\my_implementations\models\Generator", exist_ok=True)
os.makedirs(r"C:\Users\mokht\Desktop\GAN-Drug-Generator\my_implementations\models\Critic", exist_ok=True)

class Generator(nn.Module):
    
    """
    Generator class for a Wasserstein GAN with Gradient Penalty (WGAN-GP). This model generates
    data from a latent space.

    Attributes:
        embed_dim (int): Dimensionality of the input latent space.
    """
    
    def __init__(self, embed_dim: int):
        super(Generator, self).__init__()
        self.init_size = embed_dim
        # First linear layer followed by a LeakyReLU activation
        self.l1 = nn.Sequential(
            nn.Linear(self.init_size, 128), 
            nn.LeakyReLU(0.3, inplace=True)
        )
        # Stacked layers each consisting of Linear, BatchNorm1d, and LeakyReLU
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128 if i == 0 else 256, 256),
                nn.BatchNorm1d(256, 0.9),
                nn.LeakyReLU(0.3, inplace=True)
            ) for i in range(5)
        ])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass of the generator.

        Args:
            z (torch.Tensor): Input tensor, latent vector.

        Returns:
            torch.Tensor: Output tensor representing generated data.
        """
        
        out = self.l1(z)
        for layer in self.layers:
            out = layer(out)
        return out

class Discriminator(nn.Module):
    
    """
    Discriminator class for a Wasserstein GAN with Gradient Penalty (WGAN-GP). This model
    classifies the data as real or fake.

    Attributes:
        embed_dim (int): Dimensionality of the input embeddings.
    """
    
    def __init__(self, embed_dim: int):
        super(Discriminator, self).__init__()
        # Sequential model for discriminating real vs generated tensors
        self.model = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        
        """
        Forward pass of the discriminator.

        Args:
            embeddings (torch.Tensor): Input tensor representing embeddings, either real or fake.

        Returns:
            torch.Tensor: Output tensor representing the discriminator's judgment, higher values for real data.
        """
        
        return self.model(embeddings)

class WGAN_GP:
    
    """
    Implements the Wasserstein GAN with Gradient Penalty (WGAN-GP).
    This class encapsulates the generator and discriminator networks,
    training procedures, and utilities to save models.

    Attributes:
        embed_dim (int): Dimensionality of the input embeddings.
        n_critic (int): Number of discriminator updates per generator update.
        lambda_gp (float): Coefficient of gradient penalty(Lambda).
        batch_size (int): Size of each batch during training.
        n_epochs (int): Total number of epochs to train the model.
        sample_interval (int): Interval for saving the model.
        device (torch.device): Device on which to perform computations.
    """

    def __init__(self, embed_dim: int = 256, n_critic: int = 5, lambda_gp: float = 10,
                    batch_size: int = 32, n_epochs: int = 100, sample_interval: int = 400):
        self.embed_dim = embed_dim
        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.sample_interval = sample_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator(embed_dim).to(self.device)
        self.discriminator = Discriminator(embed_dim).to(self.device)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))

    def load_data(self, filepath: str) -> DataLoader:
        
        """
        Loads embeddings from a CSV file and creates a DataLoader.

        Parameters:
            filepath (str): Path to the CSV file containing embeddings.

        Returns:
            DataLoader: A DataLoader containing the embeddings.
        """
        
        try:
            df = pd.read_csv(filepath)
            embeddings = df.values
        except FileNotFoundError:
            print("CSV file not found. Please check the path and try again.")
            raise SystemExit

        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        dataset = TensorDataset(embeddings_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, dataloader: DataLoader):
        
        """
        Trains the WGAN-GP model using the provided DataLoader.

        Parameters:
            dataloader (DataLoader): DataLoader providing batches of embeddings.
        """
        
        batches_done = 0
        for epoch in range(self.n_epochs):
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{self.n_epochs}")
            for i, (embeds,) in progress_bar:
                self.train_discriminator(embeds)
                if i % self.n_critic == 0:
                    self.train_generator(embeds)

                if i % self.sample_interval == 0 or i == len(dataloader) - 1:
                    self.save_models(batches_done)
                    progress_bar.set_postfix(D_loss=self.d_loss.item(), G_loss=self.g_loss.item())
                batches_done += 1

    def train_discriminator(self, real_embeds: torch.Tensor):
        
        """
        Trains the discriminator with both real and generated embeddings.

        Parameters:
            real_embeds (torch.Tensor): Real embeddings from the DataLoader.
        """
        
        self.optimizer_D.zero_grad()
        real_embeds = real_embeds.to(self.device)
        fake_embeds = self.generate_fake(real_embeds)
        self.d_loss = self.discriminator_loss(real_embeds, fake_embeds)
        self.d_loss.backward()
        self.optimizer_D.step()

    def train_generator(self, real_embeds: torch.Tensor):
        
        """
        Trains the generator to improve its ability to fool the discriminator.

        Parameters:
            real_embeds (torch.Tensor): Real embeddings for generating fake inputs.
        """
        
        self.optimizer_G.zero_grad()
        fake_embeds = self.generate_fake(real_embeds)
        self.g_loss = self.generator_loss(fake_embeds)
        self.g_loss.backward()
        self.optimizer_G.step()

    def discriminator_loss(self, real_embeds: torch.Tensor, fake_embeds: torch.Tensor) -> torch.Tensor:
        
        """
        Calculate the loss for the discriminator using real and fake embeddings.

        Args:
            real_embeds (torch.Tensor): The real embeddings tensor from the DataLoader.
            fake_embeds (torch.Tensor): The fake embeddings tensor generated by the generator.

        Returns:
            torch.Tensor: The loss value for the discriminator.
        """
        
        real_validity = self.discriminator(real_embeds)
        fake_validity = self.discriminator(fake_embeds)
        gradient_penalty = self.compute_gradient_penalty(real_embeds, fake_embeds)
        return -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty

    def generator_loss(self, fake_embeds: torch.Tensor) -> torch.Tensor:
        
        """
        Calculate the loss for the generator based on its ability to fool the discriminator.

        Args:
            fake_embeds (torch.Tensor): The fake embeddings tensor generated by the generator.

        Returns:
            torch.Tensor: The loss value for the generator.
        """
        
        fake_validity = self.discriminator(fake_embeds)
        return -torch.mean(fake_validity)

    def generate_fake(self, real_embeds: torch.Tensor) -> torch.Tensor:
        
        """
        Generate fake embeddings using a random noise vector and the generator.

        Args:
            real_embeds (torch.Tensor): The real embeddings tensor from the DataLoader, used for dimension sizes.

        Returns:
            torch.Tensor: The fake embeddings tensor generated by the generator.
        """
        
        z = torch.tensor(np.random.uniform(-1, 1, (real_embeds.shape[0], self.embed_dim)), dtype=torch.float32, device=self.device)
        return self.generator(z)

    def compute_gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        
        """
        Calculate the gradient penalty to enforce the Lipschitz constraint on the discriminator.

        Args:
            real_samples (torch.Tensor): Real samples from the DataLoader.
            fake_samples (torch.Tensor): Fake samples generated by the generator.

        Returns:
            torch.Tensor: The gradient penalty component of the loss for the discriminator.
        """
        
        alpha = torch.tensor(np.random.random((real_samples.size(0), 1)), device=self.device, dtype=torch.float32)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones(real_samples.shape[0], 1, device=self.device, requires_grad=False)
        gradients = grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake,
                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp

    def save_models(self, batches_done: int):
        
        """
        Save the current state of the generator and discriminator models.

        Args:
            batches_done (int): The number of batches processed since the start of training. Used for file naming.
        """
        
        torch.save(self.generator.state_dict(), f"models/WGAN-GP/Generator/generator_{batches_done}.pth")
        torch.save(self.discriminator.state_dict(), f"models/WGAN-GP/Critic/discriminator_{batches_done}.pth")