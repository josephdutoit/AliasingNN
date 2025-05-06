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

class TwoLayerModel(lightning.LightningModule):
    def __init__(self, cfg, rrf=True):
        super(TwoLayerModel, self).__init__()

        self.fc1 = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)

        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.output_dim)
        self.activation = nn.ReLU()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD
        self.learning_rate = cfg.learning_rate

        if rrf:
            self.freeze_fc1()
        self.save_hyperparameters()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def freeze_fc1(self):
        for param in self.fc1.parameters():
            param.requires_grad = False
        self.fc1.weight.requires_grad = False
        self.fc1.bias.requires_grad = False
        self.fc1.weight.grad = None
        self.fc1.bias.grad = None

    