import argparse
import os
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.qm9 import QM9Dataset, collate_fn
from models.ponita import Ponita

torch.set_float32_matmul_precision('medium')


class RandomSOd(nn.Module):
        def __init__(self, d):
            """
            Initializes the RandomRotationGenerator.
            Args:
            - d (int): The dimension of the rotation matrices (2 or 3).
            """
            super(RandomSOd, self).__init__()
            assert d in [2, 3], "d must be 2 or 3."
            self.d = d

        def forward(self, n=None):
            """
            Generates random rotation matrices.
            Args:
            - n (int, optional): The number of rotation matrices to generate. If None, generates a single matrix.
            
            Returns:
            - Tensor: A tensor of shape [n, d, d] containing n rotation matrices, or [d, d] if n is None.
            """
            if self.d == 2:
                return self._generate_2d(n)
            else:
                return self._generate_3d(n)
        
        def _generate_2d(self, n):
            theta = torch.rand(n) * 2 * torch.pi if n else torch.rand(1) * 2 * torch.pi
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            rotation_matrix = torch.stack([cos_theta, -sin_theta, sin_theta, cos_theta], dim=-1)
            if n:
                return rotation_matrix.view(n, 2, 2)
            return rotation_matrix.view(2, 2)

        def _generate_3d(self, n):
            q = torch.randn(n, 4) if n else torch.randn(4)
            q = q / torch.norm(q, dim=-1, keepdim=True)
            q0, q1, q2, q3 = q.unbind(-1)
            rotation_matrix = torch.stack([
                1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2),
                2*(q1*q2 + q0*q3), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1),
                2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1**2 + q2**2)
            ], dim=-1)
            if n:
                return rotation_matrix.view(n, 3, 3)
            return rotation_matrix.view(3, 3)
        

class PONITA_QM9(pl.LightningModule):
        """
        """

        def __init__(self, args):
            super().__init__()
            self.save_hyperparameters(args)

            # For rotation augmentations during training and testing
            self.rotation_generator = RandomSOd(3)
            
            # Shift and scale before callibration
            self.shift = 0.
            self.scale = 1.

            # The metrics to log
            self.train_metric = torchmetrics.MeanAbsoluteError()
            self.valid_metric = torchmetrics.MeanAbsoluteError()
            self.test_metric = torchmetrics.MeanAbsoluteError()

            # Input/output specifications:
            in_channels_scalar = 5  # One-hot encoding
            in_channels_vec = 0  # 
            out_channels_scalar = 1  # The target
            out_channels_vec = 0  # 

            # Make the model
            self.model = Ponita(
                            input_dim         = in_channels_scalar + in_channels_vec,
                            hidden_dim        = self.hparams.hidden_dim,
                            output_dim        = out_channels_scalar,
                            num_layers        = self.hparams.layers,
                            output_dim_vec    = out_channels_vec,
                            num_ori           = self.hparams.num_ori,
                            basis_dim         = self.hparams.basis_dim,
                            degree            = self.hparams.degree,
                            widening_factor   = self.hparams.widening_factor,
                            layer_scale       = self.hparams.layer_scale,
                            task_level        = 'graph',
                            multiple_readouts = self.hparams.multiple_readouts,
                            attention         = self.hparams.attention)
        
        def set_dataset_statistics(self, dataloader):
            print('Computing dataset statistics...')
            ys = []
            for data in dataloader:
                ys.append(data['y'])
            ys = np.array(ys)
            # ys = np.concatenate(ys)
            self.shift = np.mean(ys)
            self.scale = np.std(ys)
            print('Mean and std of target are:', self.shift, '-', self.scale)

        def forward(self, batch):
            #  x, pos, edge_index, batch
            pred, _ = self.model(batch['x'], batch['pos'], batch['edge_index'], batch['batch'])
            return pred

        def training_step(self, batch):
            if self.hparams.train_augm:
                rot = self.rotation_generator().type_as(batch['pos'])
                batch['pos'] = torch.einsum('ij, bj->bi', rot, batch['pos'])
            pred = self(batch)
            loss = torch.mean((pred - (batch['y'] - self.shift) / self.scale).abs())
            self.train_metric(pred * self.scale + self.shift, batch['y'])
            return loss

        def on_train_epoch_end(self):
            self.log("train MAE", self.train_metric, prog_bar=True)

        def validation_step(self, batch, batch_idx):
            pred = self(batch)
            self.valid_metric(pred * self.scale + self.shift, batch['y'])

        def on_validation_epoch_end(self):
            self.log("valid MAE", self.valid_metric, prog_bar=True)
        
        def test_step(self, batch, batch_idx):
            pred = self(batch)
            self.test_metric(pred * self.scale + self.shift, batch['y'])

        def on_test_epoch_end(self):
            self.log("test MAE", self.test_metric, prog_bar=True)
        
        def configure_optimizers(self):
            optimizer = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        

def main(args):
    # Seed everything
    pl.seed_everything(42)

    # Load the data
    datasets = {split: QM9Dataset(split=split, target=args.target, root=args.root) for split in ['train', 'val', 'test']}
    dataloaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == 'train'), num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        for split, dataset in datasets.items()}
    
    # Hardware settings
    if args.gpus > 0:
        accelerator = "gpu"
        devices = args.gpus
    else:
        accelerator = "cpu"
        devices = "auto"
    if args.num_workers == -1:
        args.num_workers = os.cpu_count()

    # Logging settings
    if args.log:
        logger = pl.loggers.WandbLogger(project="PONITA-QM9", name=args.target.replace(" ", "_"), config=args, save_dir='logs')
    else:
        logger = None

    # Pytorch lightning call backs and trainer
    callbacks = [pl.callbacks.ModelCheckpoint(monitor='valid MAE', mode = 'min', every_n_epochs = 1, save_last=True)]
    if args.log: callbacks.append(pl.callbacks.LearningRateMonitor(logging_interval='epoch'))
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, callbacks=callbacks,
                         gradient_clip_val=0.5, accelerator=accelerator, devices=devices, enable_progress_bar=args.enable_progress_bar)

    # Do the training or testing
    if args.test_ckpt is None:
        model = PONITA_QM9(args)
        model.set_dataset_statistics(datasets['train'])
        trainer.fit(model, dataloaders['train'], dataloaders['val'], ckpt_path=args.resume_ckpt)
        trainer.test(model, dataloaders['test'], ckpt_path = "best")
    else:   
        model = PONITA_QM9.load_from_checkpoint(args.test_ckpt)
        model.set_dataset_statistics(datasets['train'])
        trainer.test(model, dataloaders['test'])


# ------------------------ Start of the main experiment script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------ Input arguments
    
    # Run parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-10)
    parser.add_argument('--log', type=eval, default=True)
    parser.add_argument('--enable_progress_bar', type=eval, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test_ckpt', type=str, default=None)
    parser.add_argument('--resume_ckpt', type=str, default=None)
    
    # Train settings
    parser.add_argument('--train_augm', type=eval, default=True)
    
    # Test settings
    parser.add_argument('--repeats', type=int, default=5)
    
    # QM9 Dataset
    parser.add_argument('--root', type=str, default="datasets/qm9")
    parser.add_argument('--target', type=str, default="alpha")
    
    # PONTA model settings
    parser.add_argument('--num_ori', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--basis_dim', type=int, default=256)
    parser.add_argument('--degree', type=int, default=2)
    parser.add_argument('--layers', type=int, default=9)
    parser.add_argument('--widening_factor', type=int, default=4)
    parser.add_argument('--layer_scale', type=eval, default=None)
    parser.add_argument('--multiple_readouts', type=eval, default=False)
    parser.add_argument('--attention', type=eval, default=False)
    
    # Parallel computing stuff
    parser.add_argument('-g', '--gpus', default=1, type=int)
    
    # Arg parser
    args = parser.parse_args()

    # Arg parser
    args = parser.parse_args()

    main(args)