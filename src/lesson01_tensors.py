import torch

def main():
    #A batch of 2 sequences each containing 4 token ids
    token_ids = make_token_ids(batch_size=2, seq_len=4, vocab_size=100)

    print("token_ids:", token_ids)
    print("shape:", token_ids.shape)
    print("dtype:", token_ids.dtype)

    #A tiny embedding table with 100 tokens and embedding dimension of 8
    embedding_table = torch.randn(100, 8)
    hidden_states = lookup_embeddings(token_ids, embedding_table)

    print("hidden_states shape:", hidden_states.shape)

    assert hidden_states.shape == (2, 4, 8)
    assert token_ids.dtype == torch.long

def make_token_ids(batch_size, seq_len, vocab_size):
    """
    Create a batch of random token ids.
    """
    return torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=torch.long)


def lookup_embeddings(token_ids, embedding_table):
    """
    Given a batch of token ids and an embedding table, return the corresponding hidden states.
    """
    return embedding_table[token_ids]



if __name__ == "__main__":
    main()
    