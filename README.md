# Token Attention Explorer

Explore single-head attention with three tokens and four-dimensional Q, K, and V vectors. Pick a query and inspect every score, weight, value vector, contribution, and output dimension.

https://raeeskasim1.github.io/token-attention-explorer/

## Calculation

1. Project handcrafted token vectors through seeded, untrained 4×4 matrices to get Q, K, and V. No biases are used.
2. Compute each query's dot product with **every** key.
3. Divide scores by √4 and the chosen temperature.
4. Apply numerically stable softmax across each row.
5. Multiply each attention weight by its **whole** corresponding V vector, then sum those vectors.

At temperature 1 this is standard scaled dot-product attention. Higher temperatures make each row more uniform.

## Scope

This is an arithmetic demonstration, not a trained NLP model. The displayed weights have no learned semantic significance. There are no positional embeddings, causal masks, residual connections, training, or multi-head layers. Rounded numbers can introduce small differences if you manually recalculate the output.

