import math
import torch
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1  # and timesteps.dtype == torch.int32

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(0, half_dim) * -emb).to(timesteps.device)
    emb = torch.matmul(1.0 * timesteps.reshape(-1, 1), emb.reshape(1, -1))
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        # emb = torch.cat([emb, torch.zeros([num_embeddings, 1])], axis=1)
        emb = F.pad(emb, [0, 1, 0, 0])
    assert list(emb.shape) == [timesteps.shape[0], embedding_dim]
    return emb
