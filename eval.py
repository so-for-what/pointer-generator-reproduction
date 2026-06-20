"""
Evaluation script for Pointer-Generator Networks.
"""
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.pointer_generator import PointerGenerator
from utils.data import build_vocab, get_dataloaders
from utils.metrics import evaluate_model


def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Build vocab
    vocab = build_vocab(
        max_vocab_size=config['model']['vocab_size'],
        max_samples=100000
    )
    print(f"Vocabulary size: {len(vocab)}")

    # Load model
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

    # Load checkpoint
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], 'best_model.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']+1})")
    else:
        print("No checkpoint found, using untrained model")

    # Get test dataloader
    _, _, test_loader = get_dataloaders(
        vocab=vocab,
        batch_size=1,
        max_enc_len=config['model']['max_enc_steps'],
        max_dec_len=config['model']['max_dec_steps'],
        max_train=0,
        max_val=0,
        max_test=config['data']['max_test_articles']
    )

    # Evaluate
    print(f"Evaluating on {len(test_loader.dataset)} test samples...")
    scores, hyps, refs = evaluate_model(
        model, test_loader, vocab, device,
        max_steps=config['model']['max_dec_steps'],
        num_samples=100  # Evaluate on first 100 for speed
    )

    # Print results
    print("\n" + "="*50)
    print("ROUGE Evaluation Results")
    print("="*50)
    for t, s in scores.items():
        print(f"  {t}:")
        print(f"    Precision: {s['precision']:.4f}")
        print(f"    Recall:    {s['recall']:.4f}")
        print(f"    F1:        {s['fmeasure']:.4f}")
    print("="*50)

    # Print sample outputs
    print("\nSample Summaries:")
    for i, (hyp, ref) in enumerate(zip(hyps, refs)):
        print(f"\n--- Sample {i+1} ---")
        print(f"Generated: {hyp}")
        print(f"Reference: {ref}")


if __name__ == '__main__':
    main()
