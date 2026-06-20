# Pointer-Generator Networks - Reproduction & Improvement

Reproduction of "Get To The Point: Summarization with Pointer-Generator Networks" (See et al., ACL 2017) in PyTorch with improvements.

## Improvement
Replace the original LSTM encoder with a **mini Transformer encoder** (2-layer) for cross-architecture comparison and analysis of copy mechanism synergy with self-attention.

## Structure
- `model/` - Model definitions (PointerGenerator, MiniTransformerEncoder)
- `utils/` - Data loading, vocab, metrics
- `configs/` - Training configuration
- `train.py` - Training script
- `eval.py` - Evaluation script (ROUGE)
- `reports/` - Report and analysis