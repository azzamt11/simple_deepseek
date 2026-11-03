import torch
from torch import nn


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        # E in R^(vocab_size x hidden_size): one vector per vocabulary token.
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        # h in R^hidden_size -> z in R^vocab_size.
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, token_ids):
        # token_ids: [batch, sequence]; each integer selects one row of E.
        hidden_states = self.embedding(token_ids)
        # hidden_states: [batch, sequence, hidden_size].
        # At each position: z = h W^T + b.
        logits = self.output(hidden_states)
        # logits: [batch, sequence, vocab_size].
        return logits


def calculate_loss(model, token_ids, targets, vocab_size):
    # logits[b, s, v] is the score for token v at position s.
    logits = model(token_ids)
    # Flatten [batch, sequence, vocab] into [predictions, vocab].
    flat_logits = logits.reshape(-1, vocab_size)
    # Flatten [batch, sequence] into one target class per prediction.
    flat_targets = targets.reshape(-1)
    # Returns the mean negative log-likelihood as one scalar.
    return nn.CrossEntropyLoss()(flat_logits, flat_targets)


def main():
    """
    This function is a complete miniature language-model training experiment.

    The important idea is not that this tiny model understands language. It
    does not. The purpose is to make the same mathematical flow used by a
    language model visible on a very small problem:

        token IDs
            -> embedding vectors
            -> hidden states
            -> vocabulary-sized logits
            -> cross-entropy loss
            -> gradients
            -> updated parameters
            -> better logits

    We create an artificial sequence containing a repeated pattern:

        4 -> 10 -> 7 -> 2 -> 4 -> 10 -> 7 -> 2

    The arrow means "should be predicted next". For example, after seeing 4,
    the desired next token is 10. After seeing 10, the desired next token is 7.

    This is called next-token prediction. At each position, the model receives
    one token and produces a complete vector of scores for every token in the
    vocabulary. If vocab_size is 100, every position produces 100 logits. The
    model is not told to output the word directly; it outputs scores, and the
    highest score can later be converted into a predicted token ID.

    Why do we remove the final input token from token_ids?
    -------------------------------------------------------
    The final token has no following token inside this short training sequence.
    There is therefore no known answer for it. We use:

        token_ids = sequence[:, :-1]
        targets   = sequence[:, 1:]

    For one row, this creates:

        input:  [4, 10, 7, 2, 4, 10, 7]
        target: [10, 7, 2, 4, 10,  7, 2]

    The input at position s is paired with the target at position s. The target
    is shifted one position to the left because it is the token that should
    come after the input token.

    What does the model calculate?
    -------------------------------
    For an input tensor with shape [batch, sequence], nn.Embedding selects a
    vector for each integer token ID. The result has shape:

        hidden_states: [batch, sequence, hidden_size]

    The linear output layer transforms every hidden vector h into logits:

        z = h W^T + b

    Here h has hidden_size values and z has vocab_size values. The model creates
    one z vector at every sequence position. The logits are raw scores, not
    probabilities. They may be negative, greater than 1, or have any real
    value.

    What does the loss calculate?
    ------------------------------
    CrossEntropyLoss compares each logits vector with the correct target ID.
    If the target is 10, the loss asks whether the score at vocabulary index 10
    is high compared with the other 99 scores. It combines softmax-like
    normalization and negative log-likelihood internally, so we pass raw logits
    directly to it and do not call softmax first.

    The model produces logits with shape:

        [batch, sequence, vocab_size]

    The loss function is easier to use after reshaping this to:

        [batch * sequence, vocab_size]

    Each row then represents one independent classification problem, and the
    flattened target tensor contains one correct class index for each row.

    What happens during one training iteration?
    ---------------------------------------------
    1. optimizer.zero_grad() removes gradients left from the previous iteration.
    2. The forward pass computes logits and a scalar loss.
    3. loss.backward() applies the chain rule and calculates how each parameter
       contributed to the loss. These derivatives are stored in parameter.grad.
    4. optimizer.step() changes each parameter using approximately:

           parameter := parameter - learning_rate * gradient

       The optimizer is trying to make the next loss smaller.

    Why repeat the loop?
    --------------------
    One update usually cannot memorize the pattern. Repeating the loop lets the
    optimizer gradually change the embedding and linear-layer parameters. The
    loss is the measurement used to tell whether the current parameters produce
    better scores for the known target tokens.

    Why can the loss be different after every run?
    ------------------------------------------------
    The model starts with randomly initialized parameters. Different random
    values can lead to different training paths. The loss should generally move
    downward for this simple repeated pattern, but it may not decrease at every
    individual step. The learning rate can be too small, too large, or simply
    cause temporary fluctuations.

    What does argmax do at the end?
    --------------------------------
    The final logits contain scores, not token IDs. `argmax(dim=-1)` selects the
    vocabulary index with the largest score at each position:

        prediction[b, s] = argmax_v(logits[b, s, v])

    This is greedy prediction. It chooses the currently highest-scoring token;
    it does not sample and it does not generate a new sequence in this lesson.

    Array illustration of argmax
    -----------------------------
    Imagine one sequence has three positions and a vocabulary of four tokens.
    The logits might be:

        logits = [
            [1.2, 4.5, 0.3, 2.0],   # position 0
            [3.1, 0.4, 5.8, 1.7],   # position 1
            [0.2, 2.2, 1.1, 6.0],   # position 2
        ]

    The row shape is [vocab_size] = [4]. Each row contains one score for:

        token ID 0, token ID 1, token ID 2, token ID 3

    At position 0, the largest value is 4.5. It is at index 1:

        argmax([1.2, 4.5, 0.3, 2.0]) = 1

    At position 1, the largest value is 5.8. It is at index 2:

        argmax([3.1, 0.4, 5.8, 1.7]) = 2

    At position 2, the largest value is 6.0. It is at index 3:

        argmax([0.2, 2.2, 1.1, 6.0]) = 3

    Therefore:

        predictions = [1, 2, 3]

    Notice that argmax returns indices, not the score values. It returns 1, 2,
    and 3, rather than 4.5, 5.8, and 6.0. Those indices are token IDs that
    can be passed to a tokenizer's decode operation to recover token text.

    With a batch, the logits have one additional outer dimension. For example,
    [batch=2, sequence=3, vocab=4] becomes predictions of shape [2, 3]:

        predictions[b, s] = argmax(logits[b, s, :])

    Why is this not yet DeepSeek-V3?
    ---------------------------------
    This model has only an embedding layer and a linear output layer. It has no
    attention, positional encoding, Transformer blocks, MLA, or DeepSeekMoE.
    Nevertheless, it already demonstrates the outer training contract shared by
    large language models: integer token IDs enter, vocabulary logits leave,
    the loss compares logits to target IDs, and backpropagation updates the
    parameters. Later lessons will improve the hidden-state computation between
    the embedding and output layers.
    """
    batch_size = 2
    sequence_length = 8
    vocab_size = 100
    hidden_size = 8

    # S has shape [batch, sequence] and contains a repeated pattern.
    sequence = torch.tensor([
        [4, 10, 7, 2, 4, 10, 7, 2],
        [4, 10, 7, 2, 4, 10, 7, 2],
    ], dtype=torch.long)

    # Teacher forcing: inputs predict the token immediately to their right.
    # token_ids: [4, 10, 7, 2, 4, 10, 7]
    # targets:   [10, 7, 2, 4, 10, 7, 2]
    token_ids = sequence[:, :-1]
    targets = sequence[:, 1:]

    model = TinyLanguageModel(vocab_size, hidden_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    for step in range(301):
        # Clear gradients from the previous iteration.
        optimizer.zero_grad()
        # Forward pass: parameters -> logits -> scalar loss.
        loss = calculate_loss(model, token_ids, targets, vocab_size)
        # Calculate d(loss)/d(parameter) for every parameter.
        loss.backward()
        # parameter := parameter - learning_rate * gradient.
        optimizer.step()

        if step % 50 == 0:
            print(f"step={step}, loss={loss.item():.4f}")

    # Fresh predictions using the trained parameters.
    final_logits = model(token_ids)
    # predictions[b, s] = argmax_v(final_logits[b, s, v]).
    predictions = final_logits.argmax(dim=-1)

    print("token_ids shape:", token_ids.shape)
    print("targets shape:", targets.shape)
    print("predictions:", predictions)
    print("targets:", targets)

    assert token_ids.shape == (batch_size, sequence_length - 1)
    assert targets.shape == (batch_size, sequence_length - 1)
    assert final_logits.shape == (batch_size, sequence_length - 1, vocab_size)
    assert predictions.shape == targets.shape


if __name__ == "__main__":
    main()
