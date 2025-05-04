import lightning
import torch
from torch import nn

class FeedForwardModel(lightning.LightningModule):
    def __init__ (self, cfg):

        num_layers = cfg.num_layers
        input_dim = cfg.input_dim
        hidden_dim = cfg.hidden_dim
        output_dim = cfg.output_dim
        activation = cfg.activation
        loss_fn = cfg.loss_fn
        optimizer = cfg.optimizer
        learning_rate = cfg.learning_rate

         # Initialize the model
        
        super(FeedForwardModel, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        if loss_fn == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_fn == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss()

        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam

        self.learning_rate = learning_rate
        self.save_hyperparameters()


    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def confgure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer
