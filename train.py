"""
Training script for Pointer-Generator Networks on CNN/Daily Mail.
"""
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pointer_generator import PointerGenerator
from utils.data import build_vocab, get_dataloaders
from utils.metrics import evaluate_model


def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    os.makedirs(config['paths']['output_dir'], exist_ok=True)

    # Build vocabulary
    vocab = build_vocab(
        max_vocab_size=config['model']['vocab_size'],
        max_samples=100000
    )

    # Get dataloaders
    train_loader, val_loader, _ = get_dataloaders(
        vocab=vocab,
        batch_size=config['training']['batch_size'],
        max_enc_len=config['model']['max_enc_steps'],
        max_dec_len=config['model']['max_dec_steps'],
        max_train=config['data']['max_train_articles'],
        max_val=config['data']['max_val_articles'],
        max_test=config['data']['max_test_articles']
    )

    # Initialize model
    model = PointerGenerator(
        vocab_size=len(vocab),
        embedding_dim=config['model']['embedding_dim'],
        hidden_dim=config['model']['hidden_dim'],
        lstm_layers=config['model']['lstm_layers'],
        dropout=config['model']['lstm_dropout'],
        coverage=config['model']['coverage'],
        max_enc_steps=config['model']['max_enc_steps'],
        max_dec_steps=config['model']['max_dec_steps'],
        pad_id=vocab.pad_id,
        sos_id=vocab.sos_id,
        eos_id=vocab.eos_id,
        unk_id=vocab.unk_id
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['lr']
    )

    # Training loop
    writer = SummaryWriter(log_dir=config['paths']['log_dir'])
    global_step = 0
    best_val_loss = float('inf')
    best_rouge = 0.0

    print(f"Starting training for {config['training']['epochs']} epochs")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_loss = 0.0
        epoch_nll = 0.0
        epoch_cov = 0.0
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            enc_input = batch['enc_input'].to(device)
            enc_mask = batch['enc_mask'].to(device)
            dec_input = batch['dec_input'].to(device)
            target = batch['target'].to(device)

            # Forward pass
            outputs = model(enc_input, enc_mask, dec_input, target)

            # Compute loss
            losses = model.compute_loss(
                outputs, target, enc_input,
                pad_id=vocab.pad_id,
                lambda_cov=1.0
            )

            # Backward pass
            optimizer.zero_grad()
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['max_grad_norm']
            )
            optimizer.step()

            # Logging
            epoch_loss += losses['loss'].item()
            epoch_nll += losses['nll_loss'].item()
            epoch_cov += losses['cov_loss'].item()
            batch_count += 1

            if batch_idx % config['training']['log_interval'] == 0:
                print(f"Epoch {epoch+1} | Step {batch_idx:5d} | "
                      f"Loss: {losses['loss'].item():.4f} | "
                      f"NLL: {losses['nll_loss'].item():.4f} | "
                      f"Cov: {losses['cov_loss'].item():.4f}")

                writer.add_scalar('train/loss', losses['loss'].item(), global_step)
                writer.add_scalar('train/nll_loss', losses['nll_loss'].item(), global_step)
                writer.add_scalar('train/cov_loss', losses['cov_loss'].item(), global_step)

            global_step += 1

            # Evaluation
            if batch_idx > 0 and batch_idx % config['training']['eval_interval'] == 0:
                val_loss = validate(model, val_loader, vocab, device)
                print(f"  Validation Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(
                        config['training']['checkpoint_dir'],
                        f'best_model.pt'
                    )
                    torch.save({
                        'epoch': epoch,
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'vocab_size': len(vocab),
                    }, checkpoint_path)
                    print(f"  Saved best model to {checkpoint_path}")

        # End of epoch
        avg_loss = epoch_loss / batch_count
        avg_nll = epoch_nll / batch_count
        avg_cov = epoch_cov / batch_count
        print(f"\nEpoch {epoch+1} summary:")
        print(f"  Avg Loss: {avg_loss:.4f} | Avg NLL: {avg_nll:.4f} | Avg Cov: {avg_cov:.4f}")

        # Save epoch checkpoint
        checkpoint_path = os.path.join(
            config['training']['checkpoint_dir'],
            f'epoch_{epoch+1}.pt'
        )
        torch.save({
            'epoch': epoch,
            'step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_loss,
            'vocab_size': len(vocab),
        }, checkpoint_path)

    print("Training complete!")
    writer.close()
    return model, vocab


def validate(model, val_loader, vocab, device):
    """Validation loop."""
    model.eval()
    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            enc_input = batch['enc_input'].to(device)
            enc_mask = batch['enc_mask'].to(device)
            dec_input = batch['dec_input'].to(device)
            target = batch['target'].to(device)

            outputs = model(enc_input, enc_mask, dec_input, target)
            losses = model.compute_loss(
                outputs, target, enc_input,
                pad_id=vocab.pad_id
            )

            total_loss += losses['loss'].item()
            count += 1

            if count >= 50:  # Validate on subset for speed
                break

    model.train()
    return total_loss / count


if __name__ == '__main__':
    config = load_config()
    model, vocab = train(config)

    # Quick evaluation
    print("\nRunning evaluation on validation set...")
    _, val_loader, _ = get_dataloaders(
        vocab=vocab,
        batch_size=1,
        max_val=100,
        max_train=0,
        max_test=0
    )

    scores, hyps, refs = evaluate_model(
        model, val_loader, vocab, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        num_samples=10
    )

    print("\nROUGE scores (sample):")
    for t, s in scores.items():
        print(f"  {t}: F1={s['fmeasure']:.4f}")

    print("\nSample outputs:")
    for i, (hyp, ref) in enumerate(zip(hyps, refs)):
        print(f"\n--- Sample {i+1} ---")
        print(f"Hyp: {hyp[:100]}...")
        print(f"Ref: {ref[:100]}...")
