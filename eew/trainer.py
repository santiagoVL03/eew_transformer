"""
Trainer module for Transformer model.

Features:
- GPU-accelerated training with mixed precision
- Class-weighted loss for imbalanced data
- Learning rate scheduling
- Early stopping
- Checkpointing
- Progress tracking
"""

import time
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .utils import AverageMeter, EarlyStopping, save_checkpoint, format_time


class Trainer:
    """
    Trainer for Transformer earthquake detection model.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        device='cuda',
        use_amp=True,
        class_weights=None,
        early_stopping_patience=10,
        checkpoint_dir='./checkpoints',
        log_interval=10,
        criterion=None,
        mixup_alpha=0.0,
        gradient_accumulation_steps=1
    ):
        """
        Args:
            model: Transformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            device: Device to train on
            use_amp: Use automatic mixed precision
            class_weights: Class weights for loss (or pos_weight for BCEWithLogitsLoss)
            early_stopping_patience: Patience for early stopping
            checkpoint_dir: Directory to save checkpoints
            log_interval: Log interval in batches
            criterion: Custom loss function (optional)
            mixup_alpha: Mixup alpha parameter (0 = disabled)
            gradient_accumulation_steps: Number of steps to accumulate gradients
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Check if using CUDA
        device_type = device.type if isinstance(device, torch.device) else device
        self.use_amp = use_amp and (device_type == 'cuda')
        self.log_interval = log_interval
        
        # Gradient accumulation
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Mixup augmentation
        self.mixup_alpha = mixup_alpha
        if mixup_alpha > 0:
            from .advanced_augmentation import MixupAugmenter
            self.mixup = MixupAugmenter(alpha=mixup_alpha, p=0.5)
        else:
            self.mixup = None
        
        # Setup loss function
        if criterion is not None:
            self.criterion = criterion
        elif class_weights is not None:
            if isinstance(class_weights, (list, tuple, np.ndarray)):
                # Assume binary classification with [weight_class0, weight_class1]
                pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
            else:
                pos_weight = torch.tensor([class_weights]).to(device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        
        # Setup AMP scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=early_stopping_patience, mode='min')
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history - store only recent epochs to save memory
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        self.max_history_size = 50  # Keep only last 50 epochs in memory (reduced from 100)
        
        # Save history to disk instead of keeping in memory
        self.history_file = self.checkpoint_dir / 'training_history.json'
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
    def _is_cuda(self):
        """Check if device is CUDA."""
        if isinstance(self.device, torch.device):
            return self.device.type == 'cuda'
        return self.device == 'cuda'
    
    def _save_history_to_disk(self):
        """Save training history to disk to free memory."""
        import json
        
        # Load existing history if available
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                disk_history = json.load(f)
        else:
            disk_history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        
        # Append current history
        for key in self.history:
            disk_history[key].extend(self.history[key])
        
        # Save to disk
        with open(self.history_file, 'w') as f:
            json.dump(disk_history, f)
        
        # Clear in-memory history after saving
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def train_epoch(self, epoch):
        """
        Train for one epoch with gradient accumulation and mixup support.
        
        Returns:
            Average training loss
        """
        self.model.train()
        
        loss_meter = AverageMeter()
        
        # DEBUG: Log start of training epoch
        logging.debug(f"Starting train_epoch {epoch}, loader has {len(self.train_loader)} batches")
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', disable=False)
        
        batch_count = 0
        accumulated_loss = 0.0
        
        for batch_idx, (waveforms, labels) in enumerate(pbar):
            batch_count += 1
            
            # Move to device (non_blocking for faster transfer)
            waveforms = waveforms.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Apply mixup augmentation if enabled
            if self.mixup is not None:
                waveforms, labels = self.mixup(waveforms, labels)
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    logits = self.model(waveforms)
                    # Clamp logits to prevent extreme values
                    logits = torch.clamp(logits, min=-20, max=20)
                    loss = self.criterion(logits, labels)
                    # Scale loss by accumulation steps
                    loss = loss / self.gradient_accumulation_steps
                
                # Check for invalid loss before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"Invalid loss detected in batch {batch_idx}, skipping...")
                    del waveforms, labels, logits, loss
                    continue
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                accumulated_loss += loss.item() * self.gradient_accumulation_steps
                
                # Update weights every N steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping to prevent explosion
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
            else:
                logits = self.model(waveforms)
                # Clamp logits to prevent extreme values
                logits = torch.clamp(logits, min=-20, max=20)
                loss = self.criterion(logits, labels)
                # Scale loss by accumulation steps
                loss = loss / self.gradient_accumulation_steps
                
                # Check for invalid loss before backward
                if torch.isnan(loss) or torch.isinf(loss):
                    logging.warning(f"Invalid loss detected in batch {batch_idx}, skipping...")
                    del waveforms, labels, logits, loss
                    continue
                
                # Backward pass
                loss.backward()
                
                accumulated_loss += loss.item() * self.gradient_accumulation_steps
                
                # Update weights every N steps
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping to prevent explosion
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            
            # Update metrics (convert to Python scalar immediately)
            loss_value = loss.item() * self.gradient_accumulation_steps
            batch_size = waveforms.size(0)
            
            # Additional safety check after .item()
            if np.isnan(loss_value) or np.isinf(loss_value):
                logging.warning(f"Invalid loss value {loss_value} in batch {batch_idx}, skipping...")
                del waveforms, labels, logits, loss
                continue
            
            loss_meter.update(loss_value, batch_size)
            
            # Explicit memory cleanup - delete tensors immediately
            del waveforms, labels, logits, loss
            
            # Clear CUDA cache more frequently for large datasets
            if self._is_cuda() and batch_idx % 20 == 0:
                torch.cuda.empty_cache()
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}'})
        
        # Handle remaining gradients if batch count not divisible by accumulation steps
        if (batch_count % self.gradient_accumulation_steps) != 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        # DEBUG: Log completion of training epoch
        
        # Final cleanup after epoch
        if self._is_cuda():
            torch.cuda.empty_cache()
        
        # Safeguard: if no batches were processed or loss is invalid, return 0.0 instead of NaN
        if loss_meter.count == 0:
            return 0.0
        
        final_loss = loss_meter.avg
        if torch.isnan(torch.tensor(final_loss)):
            return 0.0
        
        return final_loss
    
    def validate(self):
        """
        Validate model.
        
        Returns:
            val_loss, val_acc
        """
        self.model.eval()
        
        loss_meter = AverageMeter()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (waveforms, labels) in enumerate(self.val_loader):
                waveforms = waveforms.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                logits = self.model(waveforms)
                loss = self.criterion(logits, labels)
                
                # Predictions
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                # Update metrics (convert to Python scalars immediately to free GPU memory)
                loss_value = loss.item()
                batch_size = waveforms.size(0)
                correct_count = (preds == labels).sum().item()
                
                loss_meter.update(loss_value, batch_size)
                correct += correct_count
                total += batch_size
                
                # Memory cleanup - delete tensors immediately after each batch
                del waveforms, labels, logits, loss, probs, preds, loss_value, batch_size, correct_count
                
                # Clear CUDA cache periodically during validation too
                if self._is_cuda() and batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
        
        # Clear CUDA cache after validation
        if self._is_cuda():
            torch.cuda.empty_cache()
        
        val_loss = loss_meter.avg
        val_acc = correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs):
        """
        Train model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
        
        Returns:
            Training history dictionary (loaded from disk if saved)
        """
        import gc
        import json
        
        logging.info(f"Starting training for {num_epochs} epochs...")
        logging.info(f"Device: {self.device}")
        logging.info(f"Mixed precision: {self.use_amp}")
        logging.info(f"Train batches: {len(self.train_loader)}")
        logging.info(f"Val batches: {len(self.val_loader)}")
        logging.info(f"History will be saved to: {self.history_file}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Force garbage collection after training epoch
            gc.collect()
            if self._is_cuda():
                torch.cuda.empty_cache()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Force garbage collection after validation
            gc.collect()
            if self._is_cuda():
                torch.cuda.empty_cache()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Save history to disk every 5 epochs to prevent memory growth
            if epoch % 5 == 0:
                self._save_history_to_disk()
                logging.info(f"Training history saved to disk (cleared from memory)")
            
            # Log epoch results
            epoch_time = time.time() - epoch_start
            logging.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Time: {format_time(epoch_time)} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Acc: {val_acc:.4f} - "
                f"LR: {current_lr:.2e}"
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    },
                    self.checkpoint_dir / 'best_model.pth'
                )
                logging.info(f"Best model saved (val_loss: {val_loss:.4f})")
            
            # Early stopping
            if self.early_stopping(val_loss):
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Force aggressive garbage collection at end of epoch
            gc.collect()
            if self._is_cuda():
                torch.cuda.empty_cache()
                # Additional CUDA memory management
                torch.cuda.synchronize()
        
        # Save final history
        self._save_history_to_disk()
        
        total_time = time.time() - start_time
        logging.info(f"\nTraining completed in {format_time(total_time)}")
        logging.info(f"Best epoch: {self.best_epoch} (val_loss: {self.best_val_loss:.4f})")
        logging.info(f"Full training history saved to: {self.history_file}")
        
        # Load complete history from disk for return
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                complete_history = json.load(f)
            return complete_history
        else:
            return self.history


class OnlineTrainer:
    """
    Online (streaming) trainer for Transformer.
    
    In online mode, model is tested then updated on each sample/mini-batch,
    simulating real-time detection.
    """
    
    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        device='cuda',
        batch_size=1,
        update_frequency=1
    ):
        """
        Args:
            model: Transformer model
            data_loader: Data loader (entire dataset)
            optimizer: Optimizer
            device: Device
            batch_size: Batch size for online processing
            update_frequency: Update model every N samples
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.predictions = []
        self.labels_true = []
        self.timestamps = []
    
    def run(self):
        """
        Run online training/evaluation.
        
        Returns:
            predictions, true_labels, timestamps
        """
        logging.info("Starting online mode (streaming evaluation)...")
        
        self.model.train()  # Keep dropout active
        
        sample_count = 0
        
        pbar = tqdm(self.data_loader, desc='Online processing')
        
        for waveforms, labels in pbar:
            batch_size = waveforms.size(0)  # Store batch size before deletion
            waveforms = waveforms.to(self.device)
            labels = labels.to(self.device)
            
            # Test (predict before update)
            with torch.no_grad():
                logits = self.model(waveforms)
                probs = torch.sigmoid(logits)
            
            # Store predictions (convert to Python objects immediately to save memory)
            self.predictions.extend(probs.cpu().numpy().flatten().tolist())
            self.labels_true.extend(labels.cpu().numpy().flatten().tolist())
            self.timestamps.append(sample_count)
            
            # Delete intermediate tensors
            del probs
            
            # Update model
            if sample_count % self.update_frequency == 0:
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(waveforms)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                del loss
            
            # Delete batch tensors
            del waveforms, labels, logits
            
            # Clear CUDA cache periodically (more frequent for large datasets)
            if sample_count % 100 == 0:
                # Check if using CUDA
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
            
            sample_count += batch_size
            
            # Update progress
            if sample_count % 100 == 0:
                acc = np.mean((np.array(self.predictions) > 0.5) == np.array(self.labels_true))
                pbar.set_postfix({'samples': sample_count, 'acc': f'{acc:.3f}'})
        
        logging.info(f"Online processing completed: {sample_count} samples")
        
        return np.array(self.predictions), np.array(self.labels_true), np.array(self.timestamps)


if __name__ == '__main__':
    # Test trainer setup
    from .model import create_Transformer_model
    
    print("Testing trainer setup...")
    
    # Create dummy data
    train_data = [(torch.randn(8, 3, 200), torch.randint(0, 2, (8, 1)).float()) for _ in range(10)]
    val_data = [(torch.randn(8, 3, 200), torch.randint(0, 2, (8, 1)).float()) for _ in range(5)]
    
    # Create model
    model = create_Transformer_model()
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create trainer
    trainer = Trainer(
        model,
        train_data,
        val_data,
        optimizer,
        device='cpu',
        use_amp=False
    )
    
    print("Trainer created successfully!")
