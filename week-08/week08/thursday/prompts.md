# AI Usage Prompts and Critiques

### Prompt 1: Model Architecturing
**Prompt:** "Write a PyTorch LSTM class for predicting the next time step in a sequence, taking input shape (batch_size, sequence_length, hidden_dim)."
**Critique:** The AI returned a standard LSTM but required some tweaks for regression output (a single Linear layer on the last hidden state) and dropout inclusion. I adapted the architecture to ensure it correctly handled 1D time series forecasting and maintained modular shape checking.

### Prompt 2: RNN Manual Autograd
**Prompt:** "Write a manual RNN BPTT verifying PyTorch autograd gradients on a random 10-step sequence."
**Critique:** The AI returned useful code for verifying the RNN gradients. However, I had to fix the manual chain rule calculation for `dW_hh` because the AI incorrectly scaled the gradient loss term across all time steps, neglecting the sequential dependency and activation derivative (`1 - tanh^2`). I rewrote the manual backward loop to enforce `dtanh` propagation properly.

### Prompt 3: Timestamp Parsing
**Prompt:** "How to parse mixed datetime formats in pandas efficiently?"
**Critique:** The AI suggested using `dateutil` per row via `apply()`. This was extremely slow. I swapped the logic to use `pd.to_datetime(..., format='mixed')` built into pandas >= 2.0 which is faster and natively robust.
