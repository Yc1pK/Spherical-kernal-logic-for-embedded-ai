
def l2_norm(v):
    """Compute the Euclidean (L2) norm of a vector."""
    total = 0.0
    for x in v:
        total += x * x
    return total ** 0.5

def dot_product(a, b):
    """Compute the dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))

def spherical_kernel(x, y):
    """Cosine similarity kernel (unit-sphere projection, pure Python, no dependencies)."""
    x_norm = l2_norm(x)
    y_norm = l2_norm(y)
    if x_norm == 0.0 or y_norm == 0.0:
        raise ValueError("Zero vector not allowed.")
    return dot_product(x, y) / (x_norm * y_norm)

if __name__ == "__main__":
    # Example vectors
    examples = [
        ([1, 0, 0], [0, 1, 0]),
        ([1, 2, 3], [4, 5, 6]),
        ([1, 0, 0], [1, 0, 0]),
        ([1, 0, 0], [-1, 0, 0]),
    ]
    for x, y in examples:
        print(f"x = {x}, y = {y}, cosine similarity = {spherical_kernel(x, y)}")
