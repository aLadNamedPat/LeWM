"""Training script for Le World Model with W&B integration."""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from pathlib import Path
from tqdm import tqdm

from model.LWM import LWM
from model.loss import LeWMLoss


def setup_distributed():
    """Initialize distributed training if available."""
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return True, rank, local_rank, world_size
    else:
        return False, 0, 0, 1


def cleanup_distributed(is_distributed):
    """Clean up distributed training."""
    if is_distributed:
        dist.destroy_process_group()


def save_checkpoint(model, optimizer, epoch, step, loss, checkpoint_dir, is_distributed=False):
    """Save training checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get model state (unwrap DDP if needed)
    model_state = model.module.state_dict() if is_distributed else model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    # Save latest checkpoint
    latest_path = checkpoint_dir / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_path)

    # Save periodic checkpoint
    if step % 10000 == 0:
        periodic_path = checkpoint_dir / f'checkpoint_step_{step}.pt'
        torch.save(checkpoint, periodic_path)

    return latest_path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training loop."""

    # Setup distributed training
    is_distributed, rank, local_rank, world_size = setup_distributed()
    is_main_process = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process:
        print(f"Training Configuration:\n{OmegaConf.to_yaml(cfg)}")
        print(f"Using device: {device}")
        print(f"Distributed: {is_distributed} (World size: {world_size})")

    # Initialize W&B (only on main process)
    if is_main_process and wandb.run is None:
        wandb.init(
            project="le-world-model",
            name=cfg.experiment_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )

    # Set random seed
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    # Create model
    model = LWM.from_hydra_config(cfg)
    model = model.to(device)

    # Wrap with DDP if distributed
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    # Create loss function
    loss_fn = LeWMLoss(
        lambda_sigreg=cfg.loss.lambda_sigreg,
        num_projections=cfg.loss.num_projections,
        embedding_dim=cfg.model.encoder.z_dim,
    )
    loss_fn = loss_fn.to(device)  # Move loss function to same device as model

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.num_epochs,
    )

    # TODO: Create your dataloader here
    # For now, this is a placeholder
    # train_loader = create_dataloader(cfg)

    # Example placeholder dataloader
    # You'll need to replace this with your actual dataset
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1000

        def __getitem__(self, idx):
            # Return dummy data: (observations, actions)
            # observations: (N+1, C, H, W) - sequence of N+1 frames
            # actions: (N, action_dim) - N actions
            N = 5  # sequence length
            observations = torch.randn(N+1, 3, 224, 224)
            actions = torch.randn(N, cfg.model.predictor.action_dim)
            return observations, actions

    train_dataset = DummyDataset()
    sampler = DistributedSampler(train_dataset) if is_distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Resume from checkpoint if exists
    checkpoint_dir = Path(cfg.checkpoint_dir)
    latest_checkpoint = checkpoint_dir / 'checkpoint_latest.pt'
    start_epoch = 0
    global_step = 0

    if latest_checkpoint.exists():
        if is_main_process:
            print(f"Resuming from checkpoint: {latest_checkpoint}")
        start_epoch, global_step, _ = load_checkpoint(
            model.module if is_distributed else model,
            optimizer,
            latest_checkpoint,
            device
        )

    # Training loop
    if is_main_process:
        print("\nStarting training...")

    for epoch in range(start_epoch, cfg.training.num_epochs):
        if is_distributed:
            sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}") \
               if is_main_process else train_loader

        for batch_idx, (observations, actions) in enumerate(pbar):
            # Move data to device
            observations = observations.to(device)  # (B, N+1, C, H, W)
            actions = actions.to(device)  # (B, N, action_dim)

            B, N_plus_1, C, H, W = observations.shape
            N = N_plus_1 - 1

            # Encode all observations
            obs_flat = observations.view(B * N_plus_1, C, H, W)
            z_flat = model.module.encode(obs_flat) if is_distributed else model.encode(obs_flat)
            z_sequence = z_flat.view(B, N_plus_1, -1)  # (B, N+1, z_dim)

            # Split into history and targets
            z_history = z_sequence[:, :-1, :]  # (B, N, z_dim)
            z_targets = z_sequence[:, 1:, :]   # (B, N, z_dim)

            # Predict next states
            z_pred = model.module.predict(z_history, actions) if is_distributed \
                     else model.predict(z_history, actions)  # (B, N, z_dim)

            # Compute loss
            # Reshape for loss: (N, B, z_dim)
            Z_history_loss = z_history.transpose(0, 1)

            total_loss, decoder_loss, loss_dict = loss_fn(
                z_pred, z_targets, Z_history_loss
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.training.gradient_clip
            )

            optimizer.step()

            # Update metrics
            epoch_loss += total_loss.item()
            global_step += 1

            # Log to W&B (main process only)
            if is_main_process and global_step % 10 == 0:
                wandb.log({
                    "train/loss": total_loss.item(),
                    "train/loss_prediction": loss_dict['prediction'],
                    "train/loss_sigreg": loss_dict['sigreg'],
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/epoch": epoch,
                    "train/step": global_step,
                })

                pbar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'pred': f"{loss_dict['prediction']:.4f}",
                    'sigreg': f"{loss_dict['sigreg']:.4f}",
                })

            # Save checkpoint periodically
            if is_main_process and global_step % 1000 == 0:
                save_checkpoint(
                    model.module if is_distributed else model,
                    optimizer,
                    epoch,
                    global_step,
                    total_loss.item(),
                    checkpoint_dir,
                    is_distributed
                )

        # End of epoch
        scheduler.step()

        if is_main_process:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

            # Save epoch checkpoint
            save_checkpoint(
                model.module if is_distributed else model,
                optimizer,
                epoch + 1,
                global_step,
                avg_loss,
                checkpoint_dir,
                is_distributed
            )

    if is_main_process:
        print("\nTraining completed!")
        wandb.finish()

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
