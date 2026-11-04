import torch
from torch import nn

class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids):
        hidden_states = self.embedding(token_ids)
        logits = self.output(hidden_states)
        return logits

def calculate_loss(model, token_ids, targets, vocab_size):
    logits = model(token_ids)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    return nn.CrossEntropyLoss()(flat_logits, flat_targets)

def main():
    """
    This model is the tiny toy model for language word shifting model
    """
    batch_size = 2
    seq_len = 4
    vocab_size = 100
    hidden_size = 8

    sequence = torch.tensor([
        [1, 5, 7, 10, 1, 6, 7, 10],
        [1, 5, 7, 10, 1, 6, 7, 10]
    ])

    token_ids = sequence[:, :-1]
    targets = sequence[:, 1:]

    model = TinyLanguageModel(vocab_size, hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    for step in range(301):
        optimizer.zero_grad()
        loss = calculate_loss(model, token_ids, targets, vocab_size)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step = {step}, loss = {loss.item():.4f}")

    final_logits = model(token_ids)
    predictions = final_logits.argmax(dim=-1)

    print("logits.shape = ", logits.shape)
    print("final_logits.shape = ", final_logits.shape)
    print("prediction = ", prediction)

    assert final_logits.shape == (batch_size, seq_len, vocab_size)
    assert predictions.shape == (batch_size, seq_len)


if __name__ == "__main__":
    main()
