import math
import torch
import torch.nn as nn
from torch.autograd import Function

# Gradient Reversal Layer (GRL)
def grad_reverse(x, lambda_):
    return GradReverse.apply(x, lambda_)

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class SharedEncoder_linear(nn.Module):
    def __init__(self, input_dim, shared_dim, use_positional_encoding=False):
        super(SharedEncoder_linear, self).__init__()
        self.input_linear = nn.Linear(input_dim, shared_dim)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(shared_dim)
        self.ff = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),

        )
    
    def forward(self, x):
        # x: (B*T, input_dim)
        x = self.input_linear(x)
        if self.use_positional_encoding:
            x = x.unsqueeze(1)  # (B*T, 1, shared_dim)
            x = self.pos_encoder(x)
            x = x.squeeze(1)
        x = self.ff(x)
        return x

class PrivateEncoder_linear(nn.Module):
    def __init__(self, input_dim, private_dim, use_positional_encoding=False):
        super(PrivateEncoder_linear, self).__init__()
        self.input_linear = nn.Linear(input_dim, private_dim)
        self.use_positional_encoding = use_positional_encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(private_dim)
        self.ff = nn.Sequential(
            nn.Linear(private_dim, private_dim),
            nn.ReLU(),

        )
    
    def forward(self, x):
        # x: (B*T, input_dim)
        x = self.input_linear(x)
        if self.use_positional_encoding:
            x = x.unsqueeze(1)
            x = self.pos_encoder(x)
            x = x.squeeze(1)
        x = self.ff(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class DSN_Network_linear(nn.Module):
    def __init__(self, input_dim=850, shared_dim=60, private_dim=20, output_dim=2, num_domains=2,
                 grl_lambda=1.0, use_private_encoder=True, use_anchor=False):
        super(DSN_Network_linear, self).__init__()
        self.grl_lambda = grl_lambda
        self.shared_encoder = SharedEncoder_linear(input_dim, shared_dim, use_positional_encoding=False)
        self.use_private_encoder = use_private_encoder
        if self.use_private_encoder:
            self.private_encoder = PrivateEncoder_linear(input_dim, private_dim, use_positional_encoding=False)
        else:
            self.private_dummy = nn.Linear(shared_dim, private_dim)
        self.fc = nn.Linear(shared_dim, output_dim)
        if use_anchor:
            self.anchor = nn.Linear(output_dim, output_dim)
        else:
            self.anchor = None
        self.decoder = nn.Sequential(
            nn.Linear(shared_dim + private_dim, shared_dim + private_dim),
            nn.ReLU(),
            nn.Linear(shared_dim + private_dim, input_dim)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, num_domains)
        )
    
    def forward(self, x, grl_lambda=None):
        """
        x: (B, T, input_dim)
        """
        if grl_lambda is None:
            grl_lambda = self.grl_lambda
        B, T, _ = x.size()
        x_flat = x.view(B * T, -1)
        
        # Shared branch
        z_shared = self.shared_encoder(x_flat)         # (B*T, shared_dim)
        z_shared = z_shared.view(B, T, -1)               # (B, T, shared_dim)
        y_pred = self.fc(z_shared)
        
        # anchor
        if self.anchor is not None:
            anchor_pred = self.anchor(y_pred)
        else:
            anchor_pred = y_pred
        
        z_shared_mean = torch.mean(z_shared, dim=1)       # (B, shared_dim)
        
        # Private branch
        if self.use_private_encoder:
            z_private = self.private_encoder(x_flat)      # (B*T, private_dim)
            z_private = z_private.view(B, T, -1)            # (B, T, private_dim)
            z_private_mean = torch.mean(z_private, dim=1)   # (B, private_dim)
        else:
            z_private_mean = self.private_dummy(z_shared_mean)  # (B, private_dim)
        
        z_shared_rev = grad_reverse(z_shared_mean, grl_lambda)
        domain_pred = self.domain_classifier(z_shared_rev)
        
        z_combined = torch.cat([z_shared_mean, z_private_mean], dim=1)  # (B, shared_dim+private_dim)
        x_recon = self.decoder(z_combined)                              # (B, input_dim)
        x_recon = x_recon.unsqueeze(1).expand(B, T, -1)                  # (B, T, input_dim)
        
        return y_pred, x_recon, domain_pred, anchor_pred
    
    def compute_orthogonality_loss(self, z_shared, z_private):
        # z_shared, z_private: (B, latent_dim)
        return torch.mean((z_shared * z_private).pow(2))
    
