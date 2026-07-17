# CNN Architecture Design: Voice Emotion Recognition

This document provides a comprehensive overview of the Convolutional Neural Network (CNN) architecture designed for classifying vocal emotions from 2D Mel Spectrograms.

---

## 1. Core Architectural Rationale

### 1.1 Why CNNs are Suitable for Mel Spectrograms
A raw audio signal is a 1D waveform representing amplitude over time, which is highly redundant and lacks explicit frequency relations. By applying a Short-Time Fourier Transform (STFT) and mapping to a logarithmic Mel scale, the audio is transformed into a 2D Mel Spectrogram.
Mel Spectrograms exhibit **local spatial correlation** similar to digital images:
- **Vertical Axis (Frequency):** Captures spectral harmonics, pitch contours, and formants.
- **Horizontal Axis (Time):** Captures temporal progressions, rhythm, sound decays, and transitions.

Because emotional speech is characterized by distinct patterns of frequency shift (e.g. rising pitch in excitement/anger vs flat/low frequency contours in sadness) occurring over short windows, 2D CNNs are highly suited to learn these localized time-frequency structures.

### 1.2 Purpose of Individual Layers

1. **`Conv2D` (2D Convolution):** 
   Slide trainable kernel matrices across the 2D Mel Spectrogram input. Convolutions are spatially invariant, allowing them to detect local time-frequency features (like pitch glides, formant transitions, or rapid energy rises) regardless of exactly *when* in the 3-second audio clip they occur.
2. **`BatchNorm2D` (Batch Normalization):**
   Normalizes the activations of each channel across a batch to have zero mean and unit variance. This stabilizes and accelerates training by mitigating internal covariate shift. It allows for higher learning rates and acts as a minor regularizer.
3. **`ReLU` (Rectified Linear Unit):**
   Introduces non-linearity ($f(x) = \max(0, x)$), enabling the network to learn complex non-linear decision boundaries. It prevents vanishing gradient problems during backpropagation since its gradient is always $1$ for positive inputs.
4. **`MaxPool2D` (2D Max Pooling):**
   Downsamples spatial dimensions (height and width) by selecting the maximum value in local pooling windows. This reduces the number of parameters and computation, increases the receptive field of subsequent convolutional kernels, and grants translation invariance (the exact micro-position of a feature is less important than its existence).
5. **`Flatten`:**
   Flattens the high-dimensional feature maps (a 3D volume of shape `(Channels, Height, Width)`) into a single 1D vector. This allows the spatial features extracted by the CNN layers to be ingested by standard fully-connected classification layers.
6. **`Fully Connected Layer` (Linear):**
   Performs a matrix multiplication mapping the high-dimensional flattened feature vector into a lower-dimensional hidden representation. This allows the model to form global associations across all spatial feature extractions.
7. **`Dropout`:**
   A regularization technique that randomly zeroes out a fraction (e.g., $p = 0.5$) of the activations in a layer during training. This prevents the network from co-adapting weights too closely, forcing it to learn redundant representations and improving generalization to unseen speakers.
8. **Logits Output (Linear):**
   A final fully connected layer mapping hidden representations to `num_classes` outputs. These values represent raw, unnormalized prediction scores (logits) for each class.

---

## 2. Rationale for Logit-Based Outputs

The `EmotionCNN` outputs raw **logits** rather than normalized probabilities (such as from Softmax/LogSoftmax) for two primary reasons:

- **Numerical Stability in Loss Computation:** During training, PyTorch's `nn.CrossEntropyLoss` combines `log_softmax` and negative log-likelihood loss (`nn.NLLLoss`) into a single class. Mathematically, computing Softmax followed by Logarithm introduces floating-point underflow/overflow. Computing the combined loss directly from logits utilizes the log-sum-exp trick, guaranteeing numerical stability.
- **Inference Flexbility:** Normalization (e.g., Softmax) is monotonic—the index of the maximum logit matches the index of the maximum probability. For simple argmax predictions, Softmax is redundant. If probabilities are required (e.g., for multi-class thresholds or visualization), they can be computed on-demand during the inference stage without modifying the model.

---

## 3. Dynamic Flattening and Extensibility

A major design element of `EmotionCNN` is **Dynamic Flattening**. 
In standard CNN implementations, the input size of the first fully connected layer is hardcoded (e.g. `nn.Linear(22528, hidden_size)`). However, if a user changes the number of blocks, filter sizes, or pooling factors, the flattened dimension changes, resulting in shape errors.

To avoid this, `EmotionCNN` performs a dummy forward pass during `__init__` with a synthetic tensor of shape `(1, input_channels, 128, 94)`. It measures the size of the resulting feature volume (`numel()`) and uses this value to initialize the classifier dynamically. This fulfills the **Open/Closed Principle (SOLID)**, making the network architecture open to structural configuration changes without requiring manual dimension calculations.

---

## 4. Layer-by-Layer Tensor Dimensions

Below is the tensor dimension progression for the standard configuration:
- `num_classes = 8`
- `input_channels = 1`
- `filter_sizes = (32, 64, 128)`
- `kernel_sizes = (3, 3, 3)`
- `pool_sizes = (2, 2, 2)`
- `hidden_size = 256`

| Layer / Stage | Input Shape | Operation Details | Output Shape | Flattened Parameters |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | - | Mel Spectrogram Batch | `(B, 1, 128, 94)` | - |
| **Conv Block 1** | `(B, 1, 128, 94)` | Conv2D ($3\times3$, pad=1) $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ MaxPool ($2\times2$) | `(B, 32, 64, 47)` | 32 filters |
| **Conv Block 2** | `(B, 32, 64, 47)` | Conv2D ($3\times3$, pad=1) $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ MaxPool ($2\times2$) | `(B, 64, 32, 23)` | 64 filters |
| **Conv Block 3** | `(B, 64, 32, 23)` | Conv2D ($3\times3$, pad=1) $\rightarrow$ BN $\rightarrow$ ReLU $\rightarrow$ MaxPool ($2\times2$) | `(B, 128, 16, 11)` | 128 filters |
| **Flatten** | `(B, 128, 16, 11)` | Flatten spatial & channel dimensions | `(B, 22528)` | $128 \times 16 \times 11 = 22,528$ values |
| **FC Hidden** | `(B, 22528)` | Fully Connected Linear $\rightarrow$ ReLU $\rightarrow$ Dropout ($p=0.5$) | `(B, 256)` | $22,528 \times 256$ weights |
| **FC Output** | `(B, 256)` | Fully Connected Linear Output (Logits) | `(B, 8)` | $256 \times 8$ weights |

---

## 5. How the Model Learns from Mel Spectrograms

During training, the learning process unfolds through three core operations:

1. **Acoustic Texture Ingestion:** The convolutional layers learn hierarchies of features. The first block extracts low-level patterns (e.g. onset transitions, simple pitch lines). The second and third blocks extract high-level textures (e.g. pitch variance, harmonic ratio shifts, and sound decay speeds).
2. **Backpropagation of Categorical Error:** The raw logits output by the model are compared against standard target indexes via Cross-Entropy Loss. Gradients flow backward through the classification network, assigning importance to specific combinations of spectrographic features.
3. **Weight Updates:** The optimizer updates the convolutional kernel weights, tuning the network to selectively activate filters when presented with specific emotional expressions (e.g. rising pitch contours for angry voice, flat energy structures for sad voice).
