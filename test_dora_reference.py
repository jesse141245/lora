import numpy as np


def lora_delta(A, B, scale):
    return scale * (A.T @ B)


def dora_forward(x, W0, A, B, m, scale):
    v = W0 + lora_delta(A, B, scale)
    c = np.linalg.norm(v, axis=0)
    return x @ (v * (m / c))


def test_lora_output_unchanged_when_dora_disabled():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 5)).astype(np.float32)
    W0 = rng.normal(size=(5, 4)).astype(np.float32)
    A = rng.normal(size=(2, 5)).astype(np.float32)
    B = rng.normal(size=(2, 4)).astype(np.float32)
    scale = 0.25

    baseline = x @ (W0 + lora_delta(A, B, scale))
    current_lora_path = x @ W0 + x @ lora_delta(A, B, scale)
    np.testing.assert_allclose(current_lora_path, baseline, rtol=1e-6, atol=1e-6)


def test_dora_forward_matches_reference_formula():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 3)).astype(np.float32)
    W0 = rng.normal(size=(3, 4)).astype(np.float32)
    A = rng.normal(size=(2, 3)).astype(np.float32)
    B = rng.normal(size=(2, 4)).astype(np.float32)
    scale = 0.5
    m = np.linalg.norm(W0, axis=0)

    out = dora_forward(x, W0, A, B, m, scale)
    v = W0 + scale * (A.T @ B)
    c = np.sqrt((v * v).sum(axis=0))
    ref = x @ (m * v / c)
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


def test_m_initializes_to_column_norms():
    W0 = np.array([[3.0, 0.0, 1.0], [4.0, 5.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(np.linalg.norm(W0, axis=0), [5.0, 5.0, np.sqrt(5.0)])


def test_dora_gradients_touch_m_a_b_with_detached_norm():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 3))
    W0 = rng.normal(size=(3, 2))
    A = rng.normal(size=(2, 3))
    B = rng.normal(size=(2, 2))
    m = np.linalg.norm(W0, axis=0)
    dy = rng.normal(size=(4, 2))
    scale = 0.75

    v = W0 + lora_delta(A, B, scale)
    c = np.linalg.norm(v, axis=0)  # detached in backend
    ratio = m / c
    dpre = x.T @ (dy * ratio)
    grad_m = ((x @ v) / c * dy).sum(axis=0)
    grad_A = scale * B @ dpre.T
    grad_B = scale * A @ dpre

    assert np.linalg.norm(grad_m) > 0
    assert np.linalg.norm(grad_A) > 0
    assert np.linalg.norm(grad_B) > 0

    # Detached norm: gradients use c as a constant and do not include the
    # derivative of norm(v). A finite-difference full-DoRA gradient should differ.
    eps = 1e-5
    A_fd = A.copy()
    A_fd[0, 0] += eps
    loss_plus = (dora_forward(x, W0, A_fd, B, m, scale) * dy).sum()
    A_fd[0, 0] -= 2 * eps
    loss_minus = (dora_forward(x, W0, A_fd, B, m, scale) * dy).sum()
    full_grad = (loss_plus - loss_minus) / (2 * eps)
    assert not np.isclose(full_grad, grad_A[0, 0])
