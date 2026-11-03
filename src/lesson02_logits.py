import torch
from torch import nn

class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids):
        hidden_states = self.embedding(token_ids)
        print("hidden_states_shape=", hidden_states.shape)
        logits = self.output(hidden_states)
        return logits


def main():
    batch_size = 2
    seq_len = 4
    vocab_size = 50
    hidden_size = 8

    token_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
    )    

    model = TinyLanguageModel(vocab_size, hidden_size)
    logits = model(token_ids)

    print("token_ids_shape=", token_ids.shape)
    print("logits_shape=", logits.shape)

    assert logits.shape == (batch_size, seq_len, vocab_size)

if __name__ == "__main__":
    main()