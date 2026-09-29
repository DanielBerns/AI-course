w, lr = 5.0, 0.15
for step in range(8):
    grad = 2 * w            # derivative of f(w) = w**2
    w = w - lr * grad       # the update rule, eq. (11)
    print(f"step {step}:  w = {w:.4f}")
    # w -> 0.2882: convergence toward the minimum
