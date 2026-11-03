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

def main():
    batch_size = 2
    seq_len = 4
    vocab_size = 100
    hidden_size = 8

    token_ids = torch.tensor([
        [4, 10, 7, 2],
        [4, 11, 7, 2],
    ], dtype=torch.long)

    targets = torch.tensor([
        [10, 7, 2, 1],
        [11, 7, 2, 1],        
    ], dtype=torch.long) 

    model = TinyLanguageModel(vocab_size, hidden_size)
    logits = model(token_ids)

    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1)
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()

    loss.backward()

    # Gradients are available after backward() and before step().
    print("embedding gradient shape = ", model.embedding.weight.grad.shape)

    optimizer.step()

    # Recalculate predictions and loss using the updated parameters.
    updated_logits = model(token_ids)
    updated_loss = loss_function(
        updated_logits.reshape(-1, vocab_size),
        targets.reshape(-1)
    )

    print("token_ids_shape = ", token_ids.shape)
    print("logits_shape = ", logits.shape)
    print("targets_shape = ", targets.shape)
    print("loss_before_update = ", loss.item())
    print("loss_after_update = ", updated_loss.item())

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert updated_logits.shape == (batch_size, seq_len, vocab_size)
    assert targets.shape == (batch_size, seq_len)
    assert loss.ndim == 0
    assert updated_loss.ndim == 0

if __name__ == "__main__":
    main()

