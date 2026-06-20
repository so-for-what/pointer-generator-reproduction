"""
ROUGE evaluation metrics for summarization.
"""
import torch
from rouge_score import rouge_scorer
from utils.data import Vocab


class RougeEvaluator:
    """ROUGE evaluation wrapper."""

    def __init__(self, rouge_types=None):
        if rouge_types is None:
            rouge_types = ['rouge1', 'rouge2', 'rougeL']
        self.rouge_types = rouge_types
        self.scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)

    def score(self, hypothesis, reference):
        """Score a single hypothesis against reference."""
        if isinstance(hypothesis, list):
            hypothesis = ' '.join(hypothesis)
        if isinstance(reference, list):
            reference = ' '.join(reference)
        return self.scorer.score(reference, hypothesis)

    def score_batch(self, hypotheses, references):
        """Score a batch of hypotheses."""
        results = {t: {'precision': [], 'recall': [], 'fmeasure': []}
                   for t in self.rouge_types}

        for hyp, ref in zip(hypotheses, references):
            scores = self.score(hyp, ref)
            for t in self.rouge_types:
                results[t]['precision'].append(scores[t].precision)
                results[t]['recall'].append(scores[t].recall)
                results[t]['fmeasure'].append(scores[t].fmeasure)

        # Average
        avg = {}
        for t in self.rouge_types:
            avg[t] = {
                'precision': sum(results[t]['precision']) / len(results[t]['precision']),
                'recall': sum(results[t]['recall']) / len(results[t]['recall']),
                'fmeasure': sum(results[t]['fmeasure']) / len(results[t]['fmeasure']),
            }
        return avg


def evaluate_model(model, dataloader, vocab, device, max_steps=120,
                   num_samples=None):
    """Evaluate a model on a dataset using greedy decoding."""
    model.eval()
    evaluator = RougeEvaluator()

    hypotheses = []
    references = []
    count = 0

    with torch.no_grad():
        for batch in dataloader:
            if num_samples and count >= num_samples:
                break

            enc_input = batch['enc_input'].to(device)
            enc_mask = batch['enc_mask'].to(device)

            for i in range(enc_input.size(0)):
                single_input = enc_input[i:i+1]
                single_mask = enc_mask[i:i+1]

                # Greedy decode
                tokens = model.greedy_decode(single_input, single_mask, max_steps)
                hyp_text = ' '.join(vocab.decode(tokens, skip_special=True))

                # Reference
                ref_tokens = batch['target'][i].tolist()
                ref_text = ' '.join(vocab.decode(ref_tokens, skip_special=True))

                hypotheses.append(hyp_text)
                references.append(ref_text)
                count += 1

    scores = evaluator.score_batch(hypotheses, references)
    model.train()
    return scores, hypotheses[:10], references[:10]


if __name__ == '__main__':
    # Quick test
    evaluator = RougeEvaluator()
    scores = evaluator.score(
        "the cat sat on the mat",
        "the cat was sitting on the mat"
    )
    for t, s in scores.items():
        print(f"{t}: P={s.precision:.4f} R={s.recall:.4f} F={s.fmeasure:.4f}")
