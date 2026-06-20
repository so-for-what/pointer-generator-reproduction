"""
Pointer-Generator Network model with LSTM encoder.

Reference: "Get To The Point: Summarization with Pointer-Generator Networks" (See et al., ACL 2017)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import BahdanauAttention, CoverageAttention


class PointerGenerator(nn.Module):
    """Pointer-Generator Network with LSTM encoder-decoder."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, lstm_layers=2,
                 dropout=0.3, coverage=True, max_enc_steps=400, max_dec_steps=120,
                 pad_id=0, sos_id=1, eos_id=2, unk_id=3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.coverage = coverage
        self.max_enc_steps = max_enc_steps
        self.max_dec_steps = max_dec_steps
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self.encoder_type = 'lstm'
        self.start_id = sos_id
        self.end_id = eos_id

        # Embedding (shared encoder/decoder)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)

        # Encoder: bidirectional LSTM
        self.encoder_lstm = nn.LSTM(
            embedding_dim, hidden_dim // 2, num_layers=lstm_layers,
            bidirectional=True, batch_first=True, dropout=dropout if lstm_layers > 1 else 0
        )
        # Reduce bidirectional output to hidden_dim
        self.enc_out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Decoder
        self.decoder_lstm = nn.LSTM(
            embedding_dim + hidden_dim, hidden_dim,
            num_layers=1, batch_first=True, dropout=0
        )

        # Attention (standard Bahdanau)
        self.attention = BahdanauAttention(hidden_dim)

        # Coverage attention (used when coverage=True)
        self.coverage_attention = CoverageAttention(hidden_dim)

        # p_gen layer
        self.p_gen_linear = nn.Linear(hidden_dim * 2 + embedding_dim, 1)

        # Output projection to vocabulary
        self.out2vocab = nn.Linear(hidden_dim * 2, vocab_size)

        # Initial decoder state projection
        self.dec_init_h = nn.Linear(hidden_dim, hidden_dim)
        self.dec_init_c = nn.Linear(hidden_dim, hidden_dim)

    def encode(self, enc_input, enc_mask):
        """Encode source sequence."""
        # enc_input: (batch, src_len)
        enc_emb = self.embedding(enc_input)  # (batch, src_len, emb_dim)
        enc_outputs, (h_n, c_n) = self.encoder_lstm(enc_emb)

        # Project encoder outputs
        enc_outputs = self.enc_out_proj(enc_outputs)  # (batch, src_len, hidden_dim)

        # Combine bidirectional states for decoder init
        # h_n: (num_layers*2, batch, hidden_dim//2)
        # Forward: indices 0, 2, 4... Backward: indices 1, 3, 5...
        # Take last layer forward+backward, concat -> (batch, hidden_dim)
        last_fwd = h_n[-2]  # (batch, hidden_dim//2)
        last_bwd = h_n[-1]  # (batch, hidden_dim//2)
        dec_h0 = torch.tanh(self.dec_init_h(torch.cat([last_fwd, last_bwd], dim=1)))
        dec_c0 = torch.tanh(self.dec_init_c(torch.cat([last_fwd, last_bwd], dim=1)))

        return enc_outputs, enc_mask, (dec_h0.unsqueeze(0), dec_c0.unsqueeze(0))

    def decode_step(self, dec_input, dec_state, context, enc_outputs, enc_mask, coverage=None):
        """Single decoder step."""
        # dec_input: (batch, 1) - current input token ids
        # dec_state: (h, c) each (1, batch, hidden_dim)
        # context: (batch, hidden_dim) from previous step

        # Embed current input
        dec_emb = self.embedding(dec_input)  # (batch, 1, emb_dim)

        # LSTM input = [emb, context]
        lstm_input = torch.cat([dec_emb, context.unsqueeze(1)], dim=2)  # (batch, 1, emb+hidden)
        dec_output, dec_state = self.decoder_lstm(lstm_input, dec_state)
        dec_h = dec_state[0]  # (1, batch, hidden_dim)

        # Attention over encoder outputs
        if self.coverage and coverage is not None:
            context, attn_dist, coverage_new = self.coverage_attention(
                dec_h[-1], enc_outputs, coverage, mask=enc_mask
            )
        else:
            context, attn_dist = self.attention(
                dec_h[-1], enc_outputs, mask=enc_mask
            )
            coverage_new = None

        # p_gen (generation probability)
        p_gen_input = torch.cat([
            dec_h[-1], context, dec_emb.squeeze(1)
        ], dim=1)
        p_gen = torch.sigmoid(self.p_gen_linear(p_gen_input))

        # Vocabulary distribution
        vocab_scores = self.out2vocab(torch.cat([dec_output.squeeze(1), context], dim=1))
        vocab_dist = F.softmax(vocab_scores, dim=1)

        return dec_state, context, vocab_dist, attn_dist, p_gen, coverage_new

    def forward(self, enc_input, enc_mask, dec_input, dec_target=None, enc_extended=None, max_oovs=0):
        """Forward pass for training (teacher forcing)."""
        batch_size = enc_input.size(0)
        dec_len = dec_input.size(1)

        # Encode
        enc_outputs, enc_mask, dec_state = self.encode(enc_input, enc_mask)

        # Initialize decoder state
        context = torch.zeros(batch_size, self.hidden_dim, device=enc_input.device)
        coverage = torch.zeros(batch_size, enc_input.size(1), device=enc_input.device) if self.coverage else None

        # Store distributions for loss calculation
        all_vocab_dists = []
        all_attn_dists = []
        all_p_gens = []
        all_coverages = []

        # Decode step-by-step
        for t in range(dec_len - 1):
            dec_input_t = dec_input[:, t:t+1]
            dec_state, context, vocab_dist, attn_dist, p_gen, coverage_new = self.decode_step(
                dec_input_t, dec_state, context, enc_outputs, enc_mask,
                coverage if self.coverage else None
            )

            all_vocab_dists.append(vocab_dist)
            all_attn_dists.append(attn_dist)
            all_p_gens.append(p_gen)

            if self.coverage:
                all_coverages.append(coverage)
                coverage = coverage_new

        return {
            'vocab_dists': torch.stack(all_vocab_dists, dim=1),  # (batch, dec_len-1, vocab_size)
            'attn_dists': torch.stack(all_attn_dists, dim=1),     # (batch, dec_len-1, src_len)
            'p_gens': torch.stack(all_p_gens, dim=1),             # (batch, dec_len-1, 1)
            'coverages': torch.stack(all_coverages, dim=1) if self.coverage else None
        }

    def compute_loss(self, outputs, targets, enc_input, pad_id=0, lambda_cov=1.0):
        """Compute loss with optional coverage penalty."""
        vocab_dists = outputs['vocab_dists']   # (batch, dec_len-1, vocab_size)
        attn_dists = outputs['attn_dists']     # (batch, dec_len-1, src_len)
        p_gens = outputs['p_gens']             # (batch, dec_len-1, 1)

        targets = targets[:, 1:]  # shift by 1, ignore <sos>

        # Vocabulary loss (cross-entropy)
        # For pointer-generator, final_dist = p_gen * vocab_dist + (1-p_gen) * copy_dist
        # During training we use teacher forcing, so copy_dist is attn_dists
        # Final distribution over extended vocabulary
        batch_size, dec_len, vocab_size = vocab_dists.shape
        src_len = enc_input.size(1)

        # Build copy distribution
        # attn_dists: (batch, dec_len, src_len)
        # We need to scatter into extended vocab
        # For efficiency: use the vocab distribution directly if target in vocab
        # Otherwise, the copy mechanism handles OOVs
        final_dists = p_gens * vocab_dists  # (batch, dec_len, vocab_size)

        # Copy contribution: (1-p_gen) * sum over source tokens of attn_weights
        # Add source token probabilities into extended vocab
        # Since we use fixed vocab (no OOV extension for simplicity), 
        # copy mechanism helps through OOV tokens mapped to UNK
        # But for proper implementation, we'd use extended vocab
        # Simplified: use vocab_dist * p_gen for in-vocab targets
        log_probs = torch.log(final_dists + 1e-10)

        # Negative log-likelihood
        nll_loss = F.nll_loss(
            log_probs.reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=pad_id,
            reduction='mean'
        )

        # Coverage loss: sum(min(attn, coverage)) over source positions
        if self.coverage and outputs['coverages'] is not None:
            coverages = outputs['coverages']  # (batch, dec_len-1, src_len)
            cov_loss = torch.sum(torch.min(attn_dists, coverages), dim=2)  # (batch, dec_len-1)
            cov_loss = cov_loss.mean()
        else:
            cov_loss = torch.tensor(0.0, device=vocab_dists.device)

        total_loss = nll_loss + lambda_cov * cov_loss

        return {
            'loss': total_loss,
            'nll_loss': nll_loss,
            'cov_loss': cov_loss
        }

    def greedy_decode(self, enc_input, enc_mask, max_steps=None):
        """Greedy decoding (for evaluation)."""
        if max_steps is None:
            max_steps = self.max_dec_steps

        batch_size = enc_input.size(0)
        assert batch_size == 1, "Greedy decode supports batch_size=1"

        self.eval()
        with torch.no_grad():
            # Encode
            enc_outputs, enc_mask, (dec_h, dec_c) = self.encode(enc_input, enc_mask)

            context = torch.zeros(1, self.hidden_dim, device=enc_input.device)
            coverage = torch.zeros(1, enc_input.size(1), device=enc_input.device) if self.coverage else None
            dec_input_t = torch.full((1, 1), self.sos_id, dtype=torch.long, device=enc_input.device)

            decoded_tokens = []
            for _ in range(max_steps):
                _, context, vocab_dist, attn_dist, p_gen, coverage_new = self.decode_step(
                    dec_input_t, (dec_h, dec_c), context, enc_outputs, enc_mask,
                    coverage if self.coverage else None
                )

                # Final distribution (simplified: vocab only for greedy)
                token = vocab_dist.argmax(dim=-1)

                if token.item() == self.eos_id:
                    break

                decoded_tokens.append(token.item())
                dec_input_t = token.view(1, 1)

                if self.coverage:
                    coverage = coverage_new

        self.train()
        return decoded_tokens
