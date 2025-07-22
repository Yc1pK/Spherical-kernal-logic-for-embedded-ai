# z3r0sphere-ssd-logic
NO LONGER BEING BUILT ON IPHONE :D
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A lightweight, hardware-friendly **cosine-similarity** kernel that projects vectors onto the unit sphere and computes their inner product. Ideal for edge AI, firmware inference, and any resource-constrained environment.

---


## 🔍 Overview

`z3r0sphere-ssd-logic` implements the **Spherical Kernel**:
---

## 🚀 Benchmarks

On a typical ARM Cortex-M4 (single-precision float, d=128):

- **L2-normalization:** ~256 cycles
- **Dot product:** ~128 cycles
- **Total:** ~384 cycles per kernel evaluation

On modern CPUs with SIMD, throughput is much higher (see `numpy` benchmarks).

---

## 🧩 Applications

- **Edge AI**: Fast similarity search, nearest neighbor, clustering
- **Firmware inference**: On-device ML, anomaly detection
- **Signal processing**: Feature comparison, template matching
- **Information retrieval**: Embedding search, document similarity

---

## 📝 Notes

- Always check for zero vectors before normalization
- For integer/fixed-point, use scaling and lookup tables for sqrt
- For large batch, precompute norms if possible

---

# z3r0sphere-ssd-logic
CURRENTLY BEING BUILT ON IPHONE 💀
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A lightweight, hardware-friendly **cosine-similarity** kernel that projects vectors onto the unit sphere and computes their inner product. Ideal for edge AI, firmware inference, and any resource-constrained environment.

---

## 🔍 Overview

`z3r0sphere-ssd-logic` implements the **Spherical Kernel**:

\[
k(x, y) = \left\langle \frac{x}{\|x\|}, \frac{y}{\|y\|} \right\rangle
\]

This is the cosine similarity between two vectors, after projecting them onto the unit hypersphere. It is a simple, robust, and hardware-friendly kernel for comparing high-dimensional data.

---

## �️ System Process & Mathematical Logic

### 1. **Input**
Two $d$-dimensional vectors $x$ and $y$.

### 2. **L2-normalize** both vectors (project onto unit sphere)

\[
\hat{x} = \frac{x}{\|x\|_2}, \quad \hat{y} = \frac{y}{\|y\|_2}
\]

### 3. **Compute dot product**

\[
k(x, y) = \hat{x} \cdot \hat{y} = \sum_{i=1}^d \hat{x}_i \hat{y}_i
\]

### 4. **Output**
Cosine similarity in $[-1, 1]$.

#### **Properties:**
- $k(x, y) \in [-1, 1]$
- $k(x, y) = 1$ iff $x$ and $y$ are colinear and point in the same direction
- $k(x, y) = 0$ iff $x$ and $y$ are orthogonal
- $k(x, y) = -1$ iff $x$ and $y$ are colinear and point in opposite directions

#### **Why Spherical?**
Projecting onto the unit sphere removes scale, focusing only on direction. This is ideal for comparing embeddings, feature vectors, or any data where magnitude is irrelevant.

#### **Pseudocode:**
```python
def spherical_kernel(x, y):
    x_hat = x / np.linalg.norm(x)
    y_hat = y / np.linalg.norm(y)
    return np.dot(x_hat, y_hat)
```

---

## ⚙️ Features

- **Unit-sphere projection** via efficient L2-normalization
- **Cosine similarity** kernel output in $[-1,1]$
- **SIMD/NEON/Accelerator ready** — dot products map directly to hardware MACs
- **Handles high-D embeddings** with minimal numerical error
- **Reference implementations** in C (with example) and Python
- **No full kernel matrix**: just two L2-normalizations + one dot-product
- **Zero drift**: keeps all vectors on the unit hypersphere
- **Low memory footprint**: perfect for microcontrollers, DSPs, and AI-accelerator M.2 cards
- **Portable**: simple C and pure-Python implementations provided

---

## 🚀 Benchmarks

On a typical ARM Cortex-M4 (single-precision float, d=128):

- **L2-normalization:** ~256 cycles
- **Dot product:** ~128 cycles
- **Total:** ~384 cycles per kernel evaluation

On modern CPUs with SIMD, throughput is much higher (see `numpy` benchmarks).

---

## 🧩 Applications

- **Edge AI**: Fast similarity search, nearest neighbor, clustering
- **Firmware inference**: On-device ML, anomaly detection
- **Signal processing**: Feature comparison, template matching
- **Information retrieval**: Embedding search, document similarity

---

## 📝 Notes

- Always check for zero vectors before normalization
- For integer/fixed-point, use scaling and lookup tables for sqrt
- For large batch, precompute norms if possible

---

## 🧑‍💻 Reference Implementations

### Python

```python
import numpy as np

def spherical_kernel(x, y):
    """Cosine similarity kernel (unit-sphere projection)."""
    x = np.asarray(x)
    y = np.asarray(y)
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm == 0 or y_norm == 0:
        raise ValueError("Zero vector not allowed.")
    return np.dot(x, y) / (x_norm * y_norm)

# Example usage:
x = [1, 2, 3]
- **No full kernel matrix**: just two L2-normalizations + one dot-product
print(spherical_kernel(x, y))  # Output: 0.974631846
```

### C

```c
#include <math.h>
#include <stdio.h>

float spherical_kernel(const float *x, const float *y, int d) {
    float x_norm = 0.0f, y_norm = 0.0f, dot = 0.0f;
    for (int i = 0; i < d; ++i) {
        x_norm += x[i] * x[i];
        y_norm += y[i] * y[i];
        dot    += x[i] * y[i];
    }
    x_norm = sqrtf(x_norm);
    y_norm = sqrtf(y_norm);
    if (x_norm == 0.0f || y_norm == 0.0f) return 0.0f; // or handle error
    return dot / (x_norm * y_norm);
}

// Example usage:
int main() {
    float x[3] = {1, 2, 3};
    float y[3] = {4, 5, 6};
    printf("%f\n", spherical_kernel(x, y, 3)); // Output: 0.974632
    return 0;
}
```

---

## 📦 API

### Python

```python
def spherical_kernel(x: np.ndarray, y: np.ndarray) -> float:
    """Returns cosine similarity between x and y."""
```

### C

```c
float spherical_kernel(const float *x, const float *y, int d);
```

---

## 🧪 Example Inputs/Outputs

| x           | y           | Output         |
|-------------|-------------|----------------|
| [1, 0, 0]   | [0, 1, 0]   | 0.0            |
| [1, 2, 3]   | [4, 5, 6]   | 0.974631846    |
| [1, 0, 0]   | [1, 0, 0]   | 1.0            |
| [1, 0, 0]   | [-1, 0, 0]  | -1.0           |

---

## 🛠️ Quick Start

### Prerequisites

- **C**: GCC or Clang  
- **Python**: 3.7+ (and `numpy` for benchmarks)  

### Clone & Build

```bash
git clone https://github.com/Yc1pK/z3r0sphere-ssd-logic.git
cd z3r0sphere-ssd-logic
```

---

## 📚 References

- [Cosine Similarity - Wikipedia](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Kernel Methods for Pattern Analysis](https://www.kernel-machines.org/)
- [Efficient Nearest Neighbor Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search)

---

## 🏁 License

MIT License. See [LICENSE](LICENSE).
- **Zero drift**: keeps all vectors on the unit hypersphere
- **Low memory footprint**: perfect for microcontrollers, DSPs, and AI-accelerator M.2 cards
- **Portable**: simple C and pure-Python implementations provided

---

## 🧑‍💻 Reference Implementations

### Python

```python
import numpy as np

def spherical_kernel(x, y):
    """Cosine similarity kernel (unit-sphere projection)."""
    x = np.asarray(x)
    y = np.asarray(y)
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm == 0 or y_norm == 0:
        raise ValueError("Zero vector not allowed.")
    return np.dot(x, y) / (x_norm * y_norm)

# Example usage:
x = [1, 2, 3]
y = [4, 5, 6]
print(spherical_kernel(x, y))  # Output: 0.974631846
```

### C

```c
#include <math.h>
#include <stdio.h>

float spherical_kernel(const float *x, const float *y, int d) {
    float x_norm = 0.0f, y_norm = 0.0f, dot = 0.0f;
    for (int i = 0; i < d; ++i) {
        x_norm += x[i] * x[i];
        y_norm += y[i] * y[i];
        dot    += x[i] * y[i];
    }
    x_norm = sqrtf(x_norm);
    y_norm = sqrtf(y_norm);
    if (x_norm == 0.0f || y_norm == 0.0f) return 0.0f; // or handle error
    return dot / (x_norm * y_norm);
}

// Example usage:
int main() {
    float x[3] = {1, 2, 3};
    float y[3] = {4, 5, 6};
    printf("%f\n", spherical_kernel(x, y, 3)); // Output: 0.974632
    return 0;
}
```

---

## 📦 API

### Python

```python
def spherical_kernel(x: np.ndarray, y: np.ndarray) -> float:
    """Returns cosine similarity between x and y."""
```

### C

```c
float spherical_kernel(const float *x, const float *y, int d);
```

---

## 🧪 Example Inputs/Outputs

| x           | y           | Output         |
|-------------|-------------|----------------|
| [1, 0, 0]   | [0, 1, 0]   | 0.0            |
| [1, 2, 3]   | [4, 5, 6]   | 0.974631846    |
| [1, 0, 0]   | [1, 0, 0]   | 1.0            |
| [1, 0, 0]   | [-1, 0, 0]  | -1.0           |

---

## 🛠️ Quick Start

### Prerequisites

- **C**: GCC or Clang  
- **Python**: 3.7+ (and `numpy` for benchmarks)  

### Clone & Build

```bash
git clone https://github.com/Yc1pK/z3r0sphere-ssd-logic.git
cd z3r0sphere-ssd-logic
