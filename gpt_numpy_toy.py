"""A tiny trainable, character-level GPT-style model written with NumPy."""

import numpy as np


# The corpus is intentionally tiny and repeated so the example can learn it.
rng = np.random.default_rng(7)
text = "the cat sat on the mat. the cat ate a hat. the dog sat on the log. " * 80
characters = sorted(set(text))
to_id = {character: i for i, character in enumerate(characters)}
to_character = {i: character for character, i in to_id.items()}
tokens = np.array([to_id[character] for character in text])
vocab_size = len(characters)

# Small dimensions keep the NumPy matrix calculations quick.
context_size = 16
embedding_size = 16
batch_size = 16
training_steps = 1000
learning_rate = 0.05


def make_weights():
    """Create the model's randomly initialized parameters."""
    scale = 1 / np.sqrt(embedding_size)
    return {
        "token_embedding": rng.normal(0, 0.1, (vocab_size, embedding_size)),
        "position_embedding": rng.normal(0, 0.1, (context_size, embedding_size)),
        "query": rng.normal(0, scale, (embedding_size, embedding_size)),
        "key": rng.normal(0, scale, (embedding_size, embedding_size)),
        "value": rng.normal(0, scale, (embedding_size, embedding_size)),
        "attention_output": rng.normal(0, scale, (embedding_size, embedding_size)),
        "language_model_head": rng.normal(0, scale, (embedding_size, vocab_size)),
    }


weights = make_weights()
causal_mask = np.tril(np.ones((context_size, context_size), dtype=bool))


def get_batch():
    """Pick random text windows; each target is the next character."""
    last_start = len(tokens) - context_size
    starts = rng.integers(0, last_start, size=batch_size)
    inputs = np.stack([tokens[start:start + context_size] for start in starts])
    targets = np.stack([tokens[start + 1:start + context_size + 1] for start in starts])
    return inputs, targets


def softmax(values):
    """Turn a vector of scores into probabilities without overflow."""
    values = values - np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)


def forward(inputs):
    """Predict the next character at every position in each input window."""
    seq_length = inputs.shape[1]
    token_vectors = weights["token_embedding"][inputs]       # (batch, time, embed)
    position_vectors = weights["position_embedding"][:seq_length]
    x = token_vectors + position_vectors                     # broadcast over batch

    query = x @ weights["query"]
    key = x @ weights["key"]
    value = x @ weights["value"]
    scores = query @ key.transpose(0, 2, 1) / np.sqrt(embedding_size)
    allowed = causal_mask[:seq_length, :seq_length]
    scores = np.where(allowed, scores, -1e9)                  # hide future characters
    attention = softmax(scores)

    context = attention @ value
    hidden = x + context @ weights["attention_output"]       # residual connection
    logits = hidden @ weights["language_model_head"]
    cache = (x, query, key, value, attention, context, hidden, allowed)
    return logits, cache


def backward(inputs, targets, probabilities, cache):
    """Compute gradients by hand; NumPy itself does not provide autograd."""
    x, query, key, value, attention, context, hidden, allowed = cache
    batch, time = targets.shape
    grads = {name: np.zeros_like(weight) for name, weight in weights.items()}

    # Softmax-cross-entropy gradient for the next-character labels.
    dlogits = probabilities.copy()
    dlogits[np.arange(batch)[:, None], np.arange(time), targets] -= 1
    dlogits /= batch * time

    head = weights["language_model_head"]
    grads["language_model_head"] = hidden.reshape(-1, embedding_size).T @ dlogits.reshape(-1, vocab_size)
    dhidden = dlogits @ head.T

    # Backpropagate through the residual connection and attention output matrix.
    output = weights["attention_output"]
    grads["attention_output"] = context.reshape(-1, embedding_size).T @ dhidden.reshape(-1, embedding_size)
    dcontext = dhidden @ output.T
    dattention = dcontext @ value.transpose(0, 2, 1)
    dvalue = attention.transpose(0, 2, 1) @ dcontext

    # Derivative of softmax, then the query-key matrix multiplication.
    dscores = attention * (dattention - np.sum(dattention * attention, axis=-1, keepdims=True))
    dscores *= allowed[None, :, :]                             # masked scores have no gradient
    query_scale = np.sqrt(embedding_size)
    dquery = dscores @ key / query_scale
    dkey = dscores.transpose(0, 2, 1) @ query / query_scale

    # Gradients for Q, K and V projections; flatten batch and time for matrix products.
    flat_x = x.reshape(-1, embedding_size)
    grads["query"] = flat_x.T @ dquery.reshape(-1, embedding_size)
    grads["key"] = flat_x.T @ dkey.reshape(-1, embedding_size)
    grads["value"] = flat_x.T @ dvalue.reshape(-1, embedding_size)
    dx = dhidden + dquery @ weights["query"].T + dkey @ weights["key"].T + dvalue @ weights["value"].T

    # Many positions reuse the same token embedding, so add their gradients together.
    grads["position_embedding"][:time] = dx.sum(axis=0)
    np.add.at(grads["token_embedding"], inputs, dx)
    return grads


def train():
    """Train all weights with simple stochastic gradient descent."""
    for step in range(1, training_steps + 1):
        inputs, targets = get_batch()
        logits, cache = forward(inputs)
        probabilities = softmax(logits)
        chosen = probabilities[np.arange(batch_size)[:, None], np.arange(context_size), targets]
        loss = -np.log(chosen + 1e-12).mean()
        grads = backward(inputs, targets, probabilities, cache)

        for name in weights:
            weights[name] -= learning_rate * grads[name]
        if step % 200 == 0:
            print(f"step {step:4d} | loss {loss:.3f}")


def generate(prompt="the ", new_characters=120, temperature=0.8):
    """Generate text one character at a time from the learned model."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    ids = [to_id.get(character, to_id[" "]) for character in prompt]
    if not ids:
        ids = [to_id[" "]]

    for _ in range(new_characters):
        context = np.array([ids[-context_size:]])
        logits, _ = forward(context)
        next_scores = logits[0, -1] / temperature
        next_probabilities = softmax(next_scores)
        next_id = rng.choice(vocab_size, p=next_probabilities)
        ids.append(int(next_id))
    return "".join(to_character[token_id] for token_id in ids)


if __name__ == "__main__":
    train()
    print("\nGenerated text:\n" + generate())
