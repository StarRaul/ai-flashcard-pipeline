"""
Module with implementation of RNNLanguageModel class (Recurrent Neural Network Language Model).
"""

from custom_helpers import add_root_to_pythonpath
add_root_to_pythonpath(n_up=2, verbose=True)

import torch
from torch import nn
from LanguageModeling.task01_ngrams.task import BaseLanguageModel, EOS, UNK
from LanguageModeling.task05_text_tools.task import TextTools as tt


class RNNLanguageModel(nn.Module, BaseLanguageModel):
    def __init__(self, tokens: list, emb_size: int = 16, hid_size: int = 256):
        """ 
        Build a recurrent language model.
        """
        super().__init__()

        n_tokens = len(tokens)
        self.tokens = tokens

        # 1. Token embeddings
        self.embedding = nn.Embedding(n_tokens, emb_size)
        # 2. The RNN core (LSTM is the standard meta here)
        self.rnn = nn.LSTM(emb_size, hid_size, batch_first=True)
        # 3. Linear layer to predict logits (maps hid_size back to n_tokens)
        self.logits = nn.Linear(hid_size, n_tokens)

    def __call__(self, input_ix: torch.Tensor) -> torch.Tensor:
        """
        compute language model logits given input tokens
        """
        # Pass the input through the 3-layer pipeline
        embeds = self.embedding(input_ix)
        rnn_out, _ = self.rnn(embeds)  # LSTM returns a tuple of (output, hidden_states)
        logits = self.logits(rnn_out)
        return logits

    def get_possible_next_tokens(self, prefix: str) -> dict:
        """
        :returns: probabilities of next token, dict {token : prob} for all tokens
        """
        with torch.no_grad():  # Freeze gradients for inference
            # 1. Convert to matrix
            input_ix = tt.to_matrix([prefix])

            # 2. Get models output
            logits = self(input_ix)

            # 3. Apply softmax and return the result
            # Grab the very last logit in the sequence to satisfy the (1, 1, n_tokens) mock
            next_token_logits = logits[0, -1]

            probs = torch.softmax(next_token_logits, dim=-1).cpu().numpy()

        return {token: float(p) for token, p in zip(self.tokens, probs)}

    def get_next_token_prob(self, prefix: str, next_token: str) -> float:
        """ :returns: probability of next_token given prefix, float """
        return self.get_possible_next_tokens(prefix).get(next_token, 0.0)


def main():
    from LanguageModeling.task05_text_tools.task import TextTools as tt
    tokens = tt.TOKENS
    model = RNNLanguageModel(tokens, emb_size=8, hid_size=32)
    print(f"Created RNN model with {sum(p.numel() for p in model.parameters())} parameters")
    sample_text = "Hello"
    print(f"Top 3 next tokens for '{sample_text}':", 
          sorted(model.get_possible_next_tokens(sample_text).items(), key=lambda x: x[1], reverse=True)[:3])


if __name__ == '__main__':
    main()