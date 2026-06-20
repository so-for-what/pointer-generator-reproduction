"""
Attention, Copy, Coverage, Decoding layers for Pointer-Generator Networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BahdanauAttention(nn.Module):
    """
    Bahdanau additive attention for pointer-generator.
    Computes context vector as weighted sum of encoder outputs.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        Args:
            decoder_hidden: (batch, hidden_dim) - current decoder hidden state
            encoder_outputs: (batch, src_len, hidden_dim)
            mask: (batch, src_len) - padding mask, 1 for valid, 0 for pad
        Returns:
            context: (batch, hidden_dim)
            attn_weights: (batch, src_len)
        """
        # decoder_hidden: (batch, hidden_dim) -> (batch, 1, hidden_dim)
        decoder_hidden = decoder_hidden.unsqueeze(1)
        # score: (batch, src_len, 1)
        score = self.v_a(torch.tanh(self.W_a(decoder_hidden) + self.U_a(encoder_outputs)))
        attn_weights = score.squeeze(2)  # (batch, src_len)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e18)

        attn_weights = F.softmax(attn_weights, dim=1)  # (batch, src_len)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden_dim)
        return context, attn_weights


class CopyMechanism(nn.Module):
    """
    Pointer-generator copy mechanism (p_gen calculation).
    p_gen = sigmoid(W_h * h_t_dec + W_c * context + W_e * x_t_dec + b_pointer)
    """
    def __init__(self, hidden_dim, embedding_dim):
        super().__init__()
        self.gen_linear = nn.Linear(hidden_dim * 2 + embedding_dim, 1)

    def forward(self, decoder_hidden, context, dec_input):
        """
        Args:
            decoder_hidden: (batch, hidden_dim)
            context: (batch, hidden_dim)
            dec_input: (batch, embedding_dim)
        Returns:
            p_gen: (batch, 1)
        """
        x = torch.cat([decoder_hidden, context, dec_input], dim=1)
        p_gen = torch.sigmoid(self.gen_linear(x))
        return p_gen


class CoverageAttention(nn.Module):
    """
    Attention with coverage mechanism.
    Uses coverage vector to penalize repeatedly attending to same locations.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_c = nn.Linear(1, hidden_dim, bias=False)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, coverage, mask=None):
        """
        Args:
            decoder_hidden: (batch, hidden_dim)
            encoder_outputs: (batch, src_len, hidden_dim)
            coverage: (batch, src_len)
            mask: (batch, src_len)
        Returns:
            context: (batch, hidden_dim)
            attn_weights: (batch, src_len)
            coverage: updated (batch, src_len)
        """
        decoder_hidden = decoder_hidden.unsqueeze(1)  # (batch, 1, hidden_dim)
        coverage_input = coverage.unsqueeze(2)  # (batch, src_len, 1)
        score = self.v_a(torch.tanh(
            self.W_a(decoder_hidden) + self.U_a(encoder_outputs) + self.W_c(coverage_input)
        )).squeeze(2)  # (batch, src_len)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e18)

        attn_weights = F.softmax(score, dim=1)  # (batch, src_len)
        coverage = coverage + attn_weights  # update coverage

        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden_dim)
        return context, attn_weights, coverage


class BeamSearchDecoder:
    """
    Beam search decoding for pointer-generator.
    """
    def __init__(self, model, beam_size=4, max_steps=120, device='cpu'):
        self.model = model
        self.beam_size = beam_size
        self.max_steps = max_steps
        self.device = device

    def search(self, enc_input, enc_mask, oovs=None, use_coverage=False):
        """
        Args:
            enc_input: (batch, src_len)
            enc_mask: (batch, src_len)
            oovs: optional oov mapping
        Returns:
            hypotheses: list of decoded sequences
        """
        batch_size = enc_input.size(0)
        assert batch_size == 1, "Beam search currently supports batch_size=1"

        batch = enc_input.shape[0]
        # Encode
        encoder_outputs, encoder_hidden = self.model.encode(enc_input, enc_mask)

        # Initial decoder input (START token)
        dec_input = torch.full((batch, 1), self.model.start_id, dtype=torch.long, device=self.device)

        # Initial decoder hidden (from encoder final state)
        if self.model.encoder_type == 'lstm':
            dec_hidden = encoder_hidden[0][-1].unsqueeze(0).repeat(self.beam_size, 1, 1).to(self.device) if encoder_hidden is not None else None
            dec_cell = encoder_hidden[1][-1].unsqueeze(0).repeat(self.beam_size, 1, 1).to(self.device) if encoder_hidden is not None else None
        else:
            dec_hidden = torch.zeros(self.beam_size, 1, self.model.decoder.hidden_dim, device=self.device)
            dec_cell = torch.zeros(self.beam_size, 1, self.model.decoder.hidden_dim, device=self.device)

        # Expand encoder outputs for beam
        enc_outputs_expanded = encoder_outputs.repeat(self.beam_size, 1, 1)
        enc_mask_expanded = enc_mask.repeat(self.beam_size, 1)
        coverage = torch.zeros(self.beam_size, enc_mask.shape[1], device=self.device) if use_coverage else None

        # Beam search state
        hypotheses = [{'tokens': [self.model.start_id], 'score': 0.0, 'hidden': dec_hidden, 'cell': dec_cell,
                       'attn': None, 'coverage': coverage[0:1] if coverage is not None else None, 'finished': False}]

        for t in range(self.max_steps):
            all_candidates = []
            for hyp in hypotheses:
                if hyp['finished']:
                    all_candidates.append(hyp)
                    continue
                inp = torch.tensor([hyp['tokens'][-1]], device=self.device).view(1, 1)
                dec_h = hyp['hidden']
                dec_c = hyp['cell']
                cov_t = hyp['coverage']

                # Decoder step
                if self.model.encoder_type == 'lstm':
                    dec_output, (dec_h, dec_c) = self.model.decoder.lstm_step(inp, (dec_h, dec_c))
                else:
                    dec_output, (dec_h, dec_c) = self.model.decoder.lstm_step(inp, (dec_h, dec_c))

                # Attention
                if use_coverage and cov_t is not None:
                    context, attn, cov_upd = self.model.decoder.attention(enc_mask_expanded, dec_h[-1],
                                                                           enc_outputs_expanded, cov_t)
                else:
                    context, attn = self.model.decoder.attention(enc_mask_expanded, dec_h[-1], enc_outputs_expanded)
                    cov_upd = None

                # Project decoder output to vocabulary
                vocab_dist = F.softmax(self.model.decoder.out2vocab(torch.cat([dec_output, context], dim=-1)), dim=-1)

                # Copy mechanism
                if self.model.encoder_type == 'lstm':
                    p_gen = self.model.decoder.p_gen_linear(torch.cat([dec_h[-1], context, self.model.decoder.embedding(inp).squeeze(1)], dim=1))
                else:
                    p_gen = self.model.decoder.p_gen_linear(torch.cat([dec_h[-1], context, self.model.decoder.embedding(inp).squeeze(1)], dim=1))

                # Final distribution
                final_dist = torch.zeros(1, self.model.vocab_size, device=self.device)
                # Copy distribution
                copy_dist = attn.data
                # Combine
                final_dist = final_dist.scatter_add(1, enc_input.repeat(self.beam_size, 0), copy_dist)
                final_dist = (1 - p_gen) * final_dist + p_gen * vocab_dist

                # Top k
                log_probs = torch.log(final_dist + 1e-10)
                topk = log_probs.topk(self.beam_size, dim=-1)
                scores = topk[0][0]
                tokens = topk[1][0]

                for i in range(self.beam_size):
                    new_hyp = {
                        'tokens': hyp['tokens'] + [tokens[i].item()],
                        'score': hyp['score'] + scores[i].item(),
                        'hidden': dec_h,
                        'cell': dec_c,
                        'attn': None,
                        'coverage': cov_upd,
                        'finished': tokens[i].item() == self.model.end_id
                    }
                    all_candidates.append(new_hyp)

            # Prune
            all_candidates.sort(key=lambda x: x['score'] / len(x['tokens']), reverse=True)
            hypotheses = all_candidates[:self.beam_size]

            # Stop if all beams finished
            if all(h['finished'] for h in hypotheses):
                break

        # Return best
        best = hypotheses[0]
        return best['tokens'], best['score']