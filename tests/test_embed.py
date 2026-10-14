import json

import numpy as np

from research_assistant.embed import HashEmbedder


def test_hash_embedder_is_deterministic_and_similarity_follows_shared_words():
    e = HashEmbedder(dim=64)
    v = e.embed_docs(["retrieval augmented generation", "retrieval augmented models", "protein folding"])
    assert v.shape == (3, 64) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)
    assert v[0] @ v[1] > v[0] @ v[2]
    assert np.allclose(e.embed_docs(["retrieval augmented generation"])[0], v[0])
    assert e.embed_docs([]).shape == (0, 64)
    assert json.dumps(float(v[0][0]))  # plain floats, safe to serialize
