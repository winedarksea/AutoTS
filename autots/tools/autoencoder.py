# -*- coding: utf-8 -*-
"""
Autoencoder tools for anomaly detection

Variational Autoencoder (VAE) implementation for time series anomaly detection
"""
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    from torch.nn import Module, Linear, Dropout
    from torch.utils.data import DataLoader, TensorDataset
    torch_available = True
except Exception:
    from autots.tools.mocks import Module
    torch_available = False

try:
    from sklearn.preprocessing import MinMaxScaler
except Exception:
    MinMaxScaler = None

try:
    from joblib import Parallel, delayed
    joblib_present = True
except Exception:
    joblib_present = False


class VAEEncoder(Module):
    """VAE Encoder network."""
    
    def __init__(self, input_dim, latent_dim, depth=1, dropout_rate=0.0):
        super(VAEEncoder, self).__init__()
        self.depth = depth
        self.dropout_rate = dropout_rate
        
        # First layer
        layer1_dim = max(1, int(input_dim * 2 / 3))
        self.fc1 = Linear(input_dim, layer1_dim)
        
        # Second layer if depth > 1
        if depth > 1:
            layer2_dim = max(1, int(input_dim * 1 / 2))
            self.fc2 = Linear(layer1_dim, layer2_dim)
            hidden_dim = layer2_dim
        else:
            hidden_dim = layer1_dim
        
        # Latent layers
        self.fc_mean = Linear(hidden_dim, latent_dim)
        self.fc_logvar = Linear(hidden_dim, latent_dim)
        
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
        
    def forward(self, x):
        # First layer
        x = F.relu(self.fc1(x))
        if self.dropout_rate > 0:
            x = self.dropout(x)
        
        # Second layer if depth > 1
        if self.depth > 1:
            x = F.relu(self.fc2(x))
            if self.dropout_rate > 0:
                x = self.dropout(x)
        
        # Latent parameters
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        
        return mean, logvar


class VAEDecoder(Module):
    """VAE Decoder network."""
    
    def __init__(self, input_dim, latent_dim, depth=1, dropout_rate=0.0):
        super(VAEDecoder, self).__init__()
        self.depth = depth
        self.dropout_rate = dropout_rate
        
        # First layer
        layer1_dim = max(1, int(input_dim * 2 / 3))
        self.fc1 = Linear(latent_dim, layer1_dim)
        
        # Second layer if depth > 1
        if depth > 1:
            layer2_dim = max(1, int(input_dim * 1 / 2))
            self.fc2 = Linear(layer1_dim, layer2_dim)
            hidden_dim = layer2_dim
        else:
            hidden_dim = layer1_dim
        
        # Output layer
        self.fc_out = Linear(hidden_dim, input_dim)
        
        if dropout_rate > 0:
            self.dropout = Dropout(dropout_rate)
    
    def forward(self, z):
        # First layer
        x = F.relu(self.fc1(z))
        if self.dropout_rate > 0:
            x = self.dropout(x)
        
        # Second layer if depth > 1
        if self.depth > 1:
            x = F.relu(self.fc2(x))
            if self.dropout_rate > 0:
                x = self.dropout(x)
        
        # Output layer with sigmoid activation
        x = torch.sigmoid(self.fc_out(x))
        
        return x


class VAE(Module):
    """Complete VAE model."""
    
    def __init__(self, input_dim, latent_dim, depth=1, dropout_rate=0.0):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(input_dim, latent_dim, depth, dropout_rate)
        self.decoder = VAEDecoder(input_dim, latent_dim, depth, dropout_rate)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mean, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar


class VAEAnomalyDetector:
    """Variational Autoencoder for Anomaly Detection."""
    
    def __init__(self, 
                 depth=1,
                 batch_size=32,
                 epochs=50,
                 learning_rate=1e-3,
                 loss_function='elbo',
                 dropout_rate=0.0,
                 latent_dim=None,
                 beta=1.0,
                 random_state=None,
                 device=None):
        """
        Initialize VAE Anomaly Detector.
        
        Args:
            depth (int): Depth of encoder/decoder (1 or 2)
            batch_size (int): Training batch size
            epochs (int): Number of training epochs
            learning_rate (float): Adam optimizer learning rate
            loss_function (str): 'elbo', 'mse', or 'lmse'
            dropout_rate (float): Dropout rate (0.0 for no dropout)
            latent_dim (int): Latent dimension (if None, uses n*1/3)
            beta (float): Beta parameter for KL divergence weight
            random_state (int): Random seed
            device (str): Device to use ('cpu', 'cuda', or None for auto)
        """
        self.depth = depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.dropout_rate = dropout_rate
        self.latent_dim = latent_dim
        self.beta = beta
        self.random_state = random_state
        self.scaler = MinMaxScaler()
        self.model = None
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set random seeds
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
    
    def _compute_loss(self, recon_x, x, mean, logvar):
        """Compute VAE loss function."""
        # Reconstruction loss
        if self.loss_function == 'mse':
            recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        elif self.loss_function == 'lmse':
            # Log-scaled MSE
            recon_loss = F.mse_loss(torch.log(recon_x + 1e-8), torch.log(x + 1e-8), reduction='sum')
        else:  # elbo (binary cross entropy)
            recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def fit(self, X):
        """Fit the VAE model."""
        if not torch_available:
            raise ImportError("PyTorch is required for VAE anomaly detection")
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Set latent dimension if not provided
        input_dim = X_scaled.shape[1]
        if self.latent_dim is None:
            self.latent_dim = max(1, int(input_dim * 1 / 3))
        
        # Create model
        self.model = VAE(input_dim, self.latent_dim, self.depth, self.dropout_rate).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                
                recon_x, mean, logvar = self.model(batch_x)
                loss, recon_loss, kl_loss = self._compute_loss(recon_x, batch_x, mean, logvar)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
        
        return self
    
    def predict(self, X):
        """Predict anomaly scores."""
        if self.model is None:
            raise ValueError("Model must be fitted before predicting")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            recon_x, _, _ = self.model(X_tensor)
            recon_x = recon_x.cpu().numpy()
        
        # Compute reconstruction error
        if self.loss_function == 'lmse':
            reconstruction_error = np.mean((np.log(X_scaled + 1e-8) - np.log(recon_x + 1e-8)) ** 2, axis=1)
        else:
            reconstruction_error = np.mean((X_scaled - recon_x) ** 2, axis=1)
        
        return reconstruction_error


def vae_outliers(df, method_params={}):
    """VAE-based outlier detection."""
    if not torch_available:
        raise ImportError("PyTorch is required for VAE anomaly detection")
    
    # Extract parameters
    depth = method_params.get('depth', 1)
    batch_size = method_params.get('batch_size', 32)
    epochs = method_params.get('epochs', 50)
    learning_rate = method_params.get('learning_rate', 1e-3)
    loss_function = method_params.get('loss_function', 'elbo')
    dropout_rate = method_params.get('dropout_rate', 0.0)
    latent_dim = method_params.get('latent_dim', None)
    beta = method_params.get('beta', 1.0)
    contamination = method_params.get('contamination', 0.1)
    random_state = method_params.get('random_state', None)
    device = method_params.get('device', None)
    
    # Fit VAE and get reconstruction errors
    vae = VAEAnomalyDetector(
        depth=depth,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        loss_function=loss_function,
        dropout_rate=dropout_rate,
        latent_dim=latent_dim,
        beta=beta,
        random_state=random_state,
        device=device
    )
    
    vae.fit(df.values)
    scores = vae.predict(df.values)
    
    # Determine threshold based on contamination
    threshold = np.percentile(scores, (1 - contamination) * 100)
    res = np.where(scores > threshold, -1, 1)
    
    return pd.DataFrame({"anomaly": res}, index=df.index), pd.DataFrame(
        {"anomaly_score": scores}, index=df.index
    )


def loop_vae_outliers(df, method_params={}, n_jobs=1):
    """Multiprocessing on each series for multivariate VAE outliers."""
    parallel = True
    if n_jobs in [0, 1] or df.shape[1] < 5:
        parallel = False
    else:
        if not joblib_present:
            parallel = False

    # joblib multiprocessing to loop through series
    if parallel:
        df_list = Parallel(n_jobs=(n_jobs - 1))(
            delayed(vae_outliers)(
                df=df.iloc[:, i : i + 1],
                method_params=method_params,
            )
            for i in range(df.shape[1])
        )
    else:
        df_list = []
        for i in range(df.shape[1]):
            df_list.append(
                vae_outliers(
                    df=df.iloc[:, i : i + 1],
                    method_params=method_params,
                )
            )
    complete = list(map(list, zip(*df_list)))
    res = pd.concat(complete[0], axis=1)
    res.index = df.index
    res.columns = df.columns
    scores = pd.concat(complete[1], axis=1)
    scores.index = df.index
    scores.columns = df.columns
    return res, scores
