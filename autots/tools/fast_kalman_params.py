"""
Generate random state-space model parameters paired with fast_kalman.py.
"""
import random
import numpy as np

def random_state_space_original():
    """Return randomly generated statespace models."""
    n_dims = random.choices([1, 2, 3, 4, 8], [0.1, 0.2, 0.3, 0.4, 0.3])[0]
    if n_dims == 1:
        st = np.array([[1]])
        obsmod = np.random.randint(1, 3, (1, n_dims))
        procnois = np.diag(np.random.exponential(0.01, size=(n_dims)).round(3))
    else:
        st = np.random.choice([0, 1, -1], p=[0.75, 0.2, 0.05], size=(n_dims, n_dims))
        st[0, 0] = 1
        st[0, -1] = 0
        if n_dims == 2:
            obsmod = np.array([[1, 0]])
        else:
            obsmod = np.array([[1, 1] + [0] * (n_dims - 2)])
        procnois = (
            np.diag([0.2 / random.choice([1, 5, 10]), 0.001] + [0] * (n_dims - 2)) ** 2
        ).round(3)
    obsnois = random.choices(
        [1.0, 10.0, 2.0, 0.5, 0.2, 0.05, 0.001],
        [0.8, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    )[0]
    while (
        st.size > 1
        and np.all(st == 1)
        # or (st.size > 3 and np.isin(st.sum(axis=1), [0]).any())
    ):
        st, procnois, obsmod, obsnois = random_state_space()
    return st, procnois, obsmod, obsnois


def ensure_stability(st):
    eigenvalues, eigenvectors = np.linalg.eig(st)
    # Scale eigenvalues to ensure their absolute values are less than 1
    stable_eigenvalues = eigenvalues / (np.abs(eigenvalues) + 1e-5)
    st_stable = eigenvectors @ np.diag(stable_eigenvalues) @ np.linalg.inv(eigenvectors)
    return st_stable.real


def random_matrix(rows, cols, density=0.2):
    matrix = np.random.randn(rows, cols)
    sparsity_mask = np.random.rand(rows, cols) < density
    return np.where(sparsity_mask, matrix, 0)


def random_state_space(tries=15):
    for _ in range(tries):
        try:
            n_dims = random.choices([1, 2, 3, 4, 8], [0.1, 0.2, 0.3, 0.4, 0.3])[0]
            st = random_matrix(n_dims, n_dims, density=0.5)
            st = ensure_stability(st)
            obsmod = random_matrix(
                1, n_dims, density=1.0
            )  # Full observation for simplicity
            procnois = np.diag(np.random.exponential(0.01, size=n_dims)).round(3)
            obsnois = np.random.exponential(1.0)

            if np.all(np.abs(np.linalg.eigvals(st)) < 1):  # Check stability
                return st, procnois, obsmod, obsnois
        except Exception:
            pass
    # fallback
    return random_state_space_original()


def holt_winters_damped_matrices(M, alpha, beta, gamma, phi=1.0):
    """Not sure if this is correct. It's close, at least."""
    # State Transition Matrix F
    # Level & Trend Equations

    F_lt = np.array(
        [  # not sure about having the alpha and beta here
            [1 + (alpha * (1 - phi)), phi],
            [beta * (1 - phi), phi],
        ]
    )
    # Seasonal Equation
    F_s = np.eye(M, M, -1)  # This creates an identity matrix and shifts it down
    first_row = np.zeros((1, M))
    first_row[0, -1] = 1
    F_s = np.vstack([first_row, F_s[:-1]])
    F_top = np.hstack([F_lt, np.zeros((2, M))])
    F_bottom = np.hstack([np.zeros((M, 2)), F_s])
    F = np.vstack([F_top, F_bottom])
    """
    F = np.zeros((M+2, M+2))
    F[0, 0] = 1
    F[0, 1] = phi
    F[1, 0] = 1
    F[1, 1] = phi
    for i in range(2, M+2):
        F[i, i] = 1
    """

    # Process Noise Covariance Q
    Q = np.zeros((M + 2, M + 2))
    # Assuming the same variance for all states for simplicity.
    # Modify these values based on specific requirements.
    Q[0, 0] = alpha
    Q[1, 1] = beta
    for i in range(2, M + 2):
        Q[i, i] = gamma

    # Observation Matrix H
    H = np.zeros((1, M + 2))
    H[0, 0] = 1
    H[0, 1] = 1
    H[0, -1] = 1

    # Observation Noise Covariance R
    R = np.array([[1]])

    return F, Q, H, R


def _block_diag(matrices):
    """Construct a block diagonal matrix from a list of square matrices."""
    if not matrices:
        raise ValueError("No matrices provided for block diagonal construction.")
    total_dim = sum(mat.shape[0] for mat in matrices)
    result = np.zeros((total_dim, total_dim))
    offset = 0
    for mat in matrices:
        dim = mat.shape[0]
        result[offset : offset + dim, offset : offset + dim] = mat
        offset += dim
    return result


def _trend_block(trend_type, level_var, slope_var):
    """Return trend block (transition, noise, observation) for a trend type."""
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    process_noise = np.diag([level_var, slope_var])
    observation = np.array([1.0, 0.0])

    if trend_type not in {'local_linear', 'random_walk_drift'}:
        raise ValueError(f"Unsupported trend_type '{trend_type}'")

    return {'F': transition, 'Q': process_noise, 'H': observation}


def _seasonal_dummy_block(season_length, variance):
    """Create seasonal dummy block of length season_length."""
    if season_length <= 1:
        raise ValueError("season_length must be greater than 1.")

    n_states = season_length - 1
    transition = np.zeros((n_states, n_states))
    if n_states > 1:
        transition[:-1, 1:] = np.eye(n_states - 1)
    transition[-1, :] = -1
    process_noise = variance * np.eye(n_states)
    observation = np.ones(n_states)

    return {'F': transition, 'Q': process_noise, 'H': observation}


def _fourier_state_block(period, harmonics, variance, observation_phase=True):
    """Create a block for trigonometric seasonal components."""
    if harmonics <= 0:
        raise ValueError("harmonics must be a positive integer.")

    rotations = []
    obs_components = []
    for k in range(1, harmonics + 1):
        angle = 2 * np.pi * k / period
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        rotations.append(rotation)
        if observation_phase:
            obs_components.extend([np.cos(angle), np.sin(angle)])
        else:
            obs_components.extend([1.0, 0.0])

    transition = _block_diag(rotations)
    process_noise = variance * np.eye(2 * harmonics)
    observation = np.array(obs_components)

    return {'F': transition, 'Q': process_noise, 'H': observation}


def _compose_state_space(blocks):
    """Assemble state-space components from individual blocks."""
    matrices = [block['F'] for block in blocks if block is not None]
    noises = [block['Q'] for block in blocks if block is not None]
    observations = [block['H'] for block in blocks if block is not None]

    transition = _block_diag(matrices)
    process_noise = _block_diag(noises)
    observation_model = np.concatenate(observations, axis=0)

    return transition, process_noise, observation_model[np.newaxis, ...]


def new_kalman_params(method=None, allow_auto=True):
    if method in ['fast']:
        em_iter = random.choices([None, 5, 10], [0.8, 0.2, 0.1])[0]
    elif method == "superfast":
        em_iter = None
    elif method == "deep":
        em_iter = random.choices([None, 10, 20, 50, 100], [0.3, 0.6, 0.1, 0.1, 0.1])[0]
    else:
        em_iter = random.choices([None, 5, 10,], [0.3, 0.7, 0.1])[0]

    def finalize(params_dict):
        params_dict['em_iter'] = em_iter
        # Disable 'auto' observation_noise when em_iter is set to prevent
        # extremely slow nested optimization (up to 50 iterations each running em_iter EM steps)
        if em_iter is not None and params_dict.get('observation_noise') == 'auto':
            params_dict['observation_noise'] = random.choice([0.05, 0.1, 0.25])
        if not allow_auto and params_dict.get('observation_noise') == 'auto':
            params_dict['observation_noise'] = 0.1
        return params_dict

    def make_randomly_generated(original=False):
        if original:
            st, procnois, obsmod, obsnois = random_state_space_original()
            model_name = 'randomly generated_original'
        else:
            st, procnois, obsmod, obsnois = random_state_space()
            model_name = 'randomly generated'
        return {
            'model_name': model_name,
            'state_transition': st.tolist(),
            'process_noise': procnois.tolist(),
            'observation_model': obsmod.tolist(),
            'observation_noise': obsnois,
        }

    def make_dynamic_linear():
        choices = [round(v * 0.1, 1) for v in range(16)]
        return {
            'model_name': 'dynamic linear',
            'state_transition': [
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, random.choice(choices), 1],
                [0, 0, random.choice(choices), 0],
            ],
            'process_noise': [
                [random.choice(choices), 0, 0, 0],
                [0, random.choice(choices), 0, 0],
                [0, 0, random.choice(choices), 0],
                [0, 0, 0, 0],
            ],
            'observation_model': [[1, 0, 1, 0]],
            'observation_noise': 0.25,
        }

    def make_ets_aan():
        # Improved ETS (AAN) implementation with better parameter ranges
        # Alpha (level smoothing) typically 0.01 to 0.99, with emphasis on higher values
        # Beta (trend smoothing) typically 0.0 to 0.3, with emphasis on lower values
        alpha_choices = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.99]
        beta_choices = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
        
        # For ETS state space: process_noise relates to smoothing parameters
        # Smaller noise = more smoothing (closer to deterministic)
        alpha = random.choice(alpha_choices)
        beta = random.choice(beta_choices)
        
        # Convert smoothing parameters to process noise
        # For ETS: sigma_level ≈ alpha, sigma_slope ≈ beta
        level_noise = alpha
        trend_noise = beta
        
        # Observation noise should typically be small for ETS
        obs_noise_choices = [0.001, 0.01, 0.05, 0.1, 0.25, 0.5]
        
        return {
            'model_name': 'local_linear_trend_ets_aan',
            'state_transition': [[1, 1], [0, 1]],
            'process_noise': [
                [level_noise, 0.0],
                [0.0, trend_noise],
            ],
            'observation_model': [[1, 0]],
            'observation_noise': random.choice(obs_noise_choices),
        }

    def make_stochastic_dummy():
        return {
            'model_name': 'local linear stochastic seasonal dummy',
            'state_transition': [
                [1, 0, 0, 0],
                [0, -1, -1, -1],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
            ],
            'process_noise': [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            'observation_model': [[1, 1, 0, 0]],
            'observation_noise': random.choices(
                [0.25, 0.04, 0.57, 1.0, 0.08, 0.02, 0.8, 'auto'],
                [0.2, 0.3, 0.3, 0.3, 0.1, 0.05, 0.05, 0.025],
            )[0],
        }

    def make_stochastic_seasonal_7():
        return {
            'model_name': 'local linear stochastic seasonal 7',
            'state_transition': [
                [1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            'process_noise': [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            'observation_model': [[1, 0, 1, 0, 0, 0, 0, 0]],
            'observation_noise': random.choice([0.25, 'auto']),
        }

    def make_ma_model():
        return {
            'model_name': 'MA',
            'state_transition': [[1, 0], [1, 0]],
            'process_noise': [[0.2, 0.0], [0.0, 0]],
            'observation_model': [[1, 0.1]],
            'observation_noise': 1.0,
        }

    def make_ar2_model():
        return {
            'model_name': 'AR(2)',
            'state_transition': [[1, 1], [0.1, 0]],
            'process_noise': [[1, 0], [0, 0]],
            'observation_model': [[1, 0]],
            'observation_noise': 1.0,
        }

    def make_ucm_deterministic_trend():
        return {
            'model_name': 'ucm_deterministic_trend',
            'state_transition': [[1, 1], [0, 1]],
            'process_noise': [[0.01, 0], [0, 0.01]],
            'observation_model': [[1, 0]],
            'observation_noise': 0.1,
        }

    def make_ucm_random_walk_drift_ar1():
        phi = random.uniform(0.2, 0.95)
        level_noise = random.choice([0.001, 0.005, 0.01])
        drift_noise = random.choice([1e-05, 5e-05, 0.0001])
        ar_noise = random.choice([0.01, 0.05, 0.1])
        observation_noise = random.choice([0.05, 0.1, 0.2])
        return {
            'model_name': 'ucm_random_walk_drift_ar1',
            'state_transition': [
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, phi],
            ],
            'process_noise': [
                [level_noise, 0, 0],
                [0, drift_noise, 0],
                [0, 0, ar_noise],
            ],
            'observation_model': [[1, 0, 1]],
            'observation_noise': observation_noise,
            'level': 'random walk with drift',
            'cov_type': 'opg',
            'autoregressive': 1,
        }

    def make_ucm_random_walk_drift_ar1_weekly_fourier():
        phi = random.uniform(0.2, 0.95)
        level_noise = random.choice([0.001, 0.005, 0.01])
        drift_noise = random.choice([1e-05, 5e-05, 0.0001])
        ar_noise = random.choice([0.01, 0.05, 0.1])
        sigma_fourier2 = random.choice([5e-3, 1e-2])
        harmonics = random.choices([2, 3, 4], [0.4, 0.4, 0.2])[0]

        trend_block = _trend_block(
            trend_type='random_walk_drift',
            level_var=level_noise,
            slope_var=drift_noise,
        )
        ar_block = {'F': np.array([[phi]]), 'Q': np.array([[ar_noise]]), 'H': np.array([1.0])}
        fourier_block = _fourier_state_block(
            period=7, harmonics=harmonics, variance=sigma_fourier2, observation_phase=False
        )

        F_full, Q_block, H_full = _compose_state_space(
            [trend_block, ar_block, fourier_block]
        )

        return {
            'model_name': 'ucm_random_walk_drift_ar1_weekly_fourier',
            'state_transition': F_full.tolist(),
            'process_noise': Q_block.tolist(),
            'observation_model': H_full.tolist(),
            'observation_noise': random.choice([0.03, 0.05, 0.1, 'auto']),
            'level': 'random walk with drift',
            'cov_type': 'opg',
            'autoregressive': 1,
        }

    def make_x1_model():
        return {
            'model_name': 'X1',
            'state_transition': [[1, 1, 0], [0, 1, 0], [0, 0, 1]],
            'process_noise': [
                [0.1, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.0, 0.0, 0.1],
            ],
            'observation_model': [[1, 1, 1]],
            'observation_noise': 1.0,
        }

    def make_hidden_state_seasonal_7():
        return {
            'model_name': "local linear hidden state with seasonal 7",
            'state_transition': [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ],
            'process_noise': [
                [0.0016, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            'observation_model': [[1, 1, 0, 0, 0, 0, 0, 0]],
            'observation_noise': random.choice([0.25, 0.5, 1.0, 0.04, 0.02, 'auto']),
        }

    def make_factor_model():
        return {
            'model_name': "factor",
            'state_transition': [
                [1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            'process_noise': [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            'observation_model': [[1, 0, 0, 0, 0, 0]],
            'observation_noise': 0.04,
        }

    def make_ucm_deterministictrend_seasonal7():
        return {
            'model_name': "ucm_deterministictrend_seasonal7",
            'state_transition': [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, -1, -1, -1, -1, -1, -1],
            ],
            'process_noise': [
                [0.001, 0, 0, 0, 0, 0, 0, 0],
                [0, 0.001, 0, 0, 0, 0, 0, 0],
                [0, 0, 0.001, 0, 0, 0, 0, 0],
                [0, 0, 0, 0.001, 0, 0, 0, 0],
                [0, 0, 0, 0, 0.001, 0, 0, 0],
                [0, 0, 0, 0, 0, 0.001, 0, 0],
                [0, 0, 0, 0, 0, 0, 0.001, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ],
            'observation_model': [[1, 0, 1, 1, 1, 1, 1, 1]],
            'observation_noise': random.choice([0.03, 'auto']),
        }

    def make_spline_model():
        return {
            'model_name': 'spline',
            'state_transition': [[2, -1], [1, 0]],
            'process_noise': [[1, 0], [0, 0]],
            'observation_model': [[1, 0]],
            'observation_noise': random.choice([0.1, 'auto']),
        }

    def make_locallinear_weekly_fourier():
        K_val = random.choices([2, 3, 6, 8], [0.1, 0.6, 0.1, 0.1])[0]
        sigma_level2 = 1e-2
        sigma_slope2 = 1e-3
        sigma_weekly2 = 1e-2
        sigma_fourier2 = 1e-2

        trend_block = _trend_block(
            trend_type='local_linear', level_var=sigma_level2, slope_var=sigma_slope2
        )
        weekly_block = _seasonal_dummy_block(season_length=7, variance=sigma_weekly2)
        fourier_block = _fourier_state_block(
            period=365.25, harmonics=K_val, variance=sigma_fourier2
        )

        F_full, Q_block, H_full = _compose_state_space(
            [trend_block, weekly_block, fourier_block]
        )

        return {
            'model_name': 'locallinear_weekly_fourier',
            'state_transition': F_full.tolist(),
            'process_noise': Q_block.tolist(),
            'observation_model': H_full.tolist(),
            'observation_noise': random.choice([0.01, 0.1, 'auto']),
        }

    def make_locallinear_daily_fourier():
        K_val = random.choices([2, 4, 6, 8], [0.2, 0.4, 0.2, 0.2])[0]
        sigma_level2 = random.choice([5e-3, 1e-2])
        sigma_slope2 = random.choice([5e-4, 1e-3])
        sigma_season2 = random.choice([5e-3, 1e-2])
        sigma_fourier2 = random.choice([5e-3, 1e-2])

        trend_block = _trend_block(
            trend_type='local_linear', level_var=sigma_level2, slope_var=sigma_slope2
        )
        daily_block = _seasonal_dummy_block(season_length=24, variance=sigma_season2)
        fourier_block = _fourier_state_block(
            period=24, harmonics=K_val, variance=sigma_fourier2, observation_phase=False
        )

        F_full, Q_block, H_full = _compose_state_space(
            [trend_block, daily_block, fourier_block]
        )

        return {
            'model_name': 'locallinear_daily_fourier',
            'state_transition': F_full.tolist(),
            'process_noise': Q_block.tolist(),
            'observation_model': H_full.tolist(),
            'observation_noise': random.choice([0.01, 0.1, 'auto']),
        }

    def make_randomwalk_weekly_fourier():
        K_val = random.choices([2, 3, 4], [0.3, 0.5, 0.2])[0]
        sigma_level2 = random.choice([5e-3, 1e-2, 5e-2])
        sigma_drift2 = random.choice([1e-4, 5e-4, 1e-3])
        sigma_fourier2 = random.choice([5e-3, 1e-2])

        trend_block = _trend_block(
            trend_type='random_walk_drift',
            level_var=sigma_level2,
            slope_var=sigma_drift2,
        )
        fourier_block = _fourier_state_block(
            period=7, harmonics=K_val, variance=sigma_fourier2, observation_phase=False
        )

        F_full, Q_block, H_full = _compose_state_space([trend_block, fourier_block])

        return {
            'model_name': 'randomwalk_weekly_fourier',
            'state_transition': F_full.tolist(),
            'process_noise': Q_block.tolist(),
            'observation_model': H_full.tolist(),
            'observation_noise': random.choice([0.01, 0.1, 'auto']),
        }

    def make_holt_winters_damped():
        F_hw, Q_hw, H_hw, _ = holt_winters_damped_matrices(
            M=7,
            alpha=random.choice([0.9, 1.0, 0, 0.3]),
            beta=random.choice([0.9, 1.0, 0, 0.3]),
            gamma=random.choice([0.9, 1.0, 0, 0.3]),
            phi=random.choice([0.9, 1.0, 0.995, 0.98]),
        )
        return {
            'model_name': 'holt_winters_damped',
            'state_transition': F_hw.tolist(),
            'process_noise': Q_hw.tolist(),
            'observation_model': H_hw.tolist(),
            'observation_noise': random.choice([0.01, 0.1, 1, 'auto']),
        }

    def make_seasonal_hidden_state(length):
        season_length = length
        if season_length == 364 and method not in ['deep']:
            season_length = 7
        state_transition = np.zeros((season_length + 1, season_length + 1))
        state_transition[0, 0] = 1
        state_transition[1, 1:-1] = [-1.0] * (season_length - 1)
        state_transition[2:, 1:-1] = np.eye(season_length - 1)
        observation_model = [[1, 1] + [0] * (season_length - 1)]
        level_noise = 0.2 / random.choice([0.2, 0.5, 1, 5, 10, 200])
        season_noise = random.choice([1e-4, 1e-3, 1e-2, 1e-1])
        process_noise_cov = (
            np.diag([level_noise, season_noise] + [0] * (season_length - 1)) ** 2
        )
        return {
            'model_name': f'local linear hidden state with seasonal {season_length}',
            'state_transition': state_transition.tolist(),
            'process_noise': process_noise_cov.tolist(),
            'observation_model': observation_model,
            'observation_noise': 0.04,
        }

    def make_theta_equivalent():
        # State-space equivalent of the Theta method (SES with drift).
        # Use smaller variances for smoother forecasts
        level_variance = random.choice([1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
        observation_variance = random.choice([0.001, 0.01, 0.05, 0.1, 0.5])
        return {
            'model_name': 'theta_equivalent',
            'state_transition': [[1, 1], [0, 1]],
            'process_noise': [[level_variance, 0], [0, 0]],
            'observation_model': [[1, 0]],
            'observation_noise': observation_variance,
        }

    def make_ets_simple_exponential_smoothing():
        # Simple Exponential Smoothing (SES) - ETS(A,N,N)
        # Only level, no trend or seasonality
        alpha = random.choice([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        return {
            'model_name': 'ets_simple_exponential_smoothing',
            'state_transition': [[1]],
            'process_noise': [[alpha]],
            'observation_model': [[1]],
            'observation_noise': random.choice([0.001, 0.01, 0.05, 0.1]),
        }

    def make_ets_linear_trend():
        # Linear Trend (Holt) - ETS(A,A,N)  
        alpha = random.choice([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        beta = random.choice([0.001, 0.01, 0.05, 0.1, 0.2])
        return {
            'model_name': 'ets_linear_trend',
            'state_transition': [[1, 1], [0, 1]],
            'process_noise': [[alpha, 0.0], [0.0, beta]],
            'observation_model': [[1, 0]],
            'observation_noise': random.choice([0.001, 0.01, 0.05, 0.1]),
        }

    def make_ets_damped_trend():
        # Damped Trend - ETS(A,Ad,N)
        alpha = random.choice([0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        beta = random.choice([0.001, 0.01, 0.05, 0.1])
        phi = random.choice([0.8, 0.85, 0.9, 0.95, 0.98])  # damping parameter
        return {
            'model_name': 'ets_damped_trend',
            'state_transition': [[1, phi], [0, phi]],
            'process_noise': [[alpha, 0.0], [0.0, beta]],
            'observation_model': [[1, 1]],
            'observation_noise': random.choice([0.001, 0.01, 0.05, 0.1]),
        }

    def make_arima_model():
        # ARIMA(p,0,q) in state space form - favoring (1,0,4) but allowing variations
        # Choose AR order (p) - favor p=1
        p = random.choices([0, 1, 2, 3], weights=[0.1, 0.6, 0.2, 0.1])[0]
        
        # Choose MA order (q) - favor q=4
        q = random.choices([0, 1, 2, 3, 4, 5, 6], weights=[0.05, 0.1, 0.1, 0.1, 0.4, 0.15, 0.1])[0]
        
        # Ensure at least one of p or q is non-zero
        if p == 0 and q == 0:
            p = 1
            q = 4
        
        # Generate AR coefficients (ensuring stationarity)
        ar_coeffs = []
        if p > 0:
            for i in range(p):
                # First AR coefficient can be larger, subsequent ones smaller
                if i == 0:
                    ar_coeffs.append(random.choice([0.1, 0.3, 0.5, 0.7, 0.9, 0.95]))
                else:
                    ar_coeffs.append(random.choice([0.0, 0.1, 0.2, 0.3]))
        
        # Generate MA coefficients (decreasing magnitude for higher orders)
        ma_coeffs = []
        if q > 0:
            for i in range(q):
                if i == 0:
                    ma_coeffs.append(random.choice([0.1, 0.3, 0.5, 0.7, 0.9]))
                elif i == 1:
                    ma_coeffs.append(random.choice([0.0, 0.1, 0.3, 0.5]))
                else:
                    ma_coeffs.append(random.choice([0.0, 0.1, 0.3]))
        
        # For ARMA(p,q) state space, dimension is max(p, q+1)
        # State vector: [y_t, y_{t-1}, ..., y_{t-p+1}, ε_{t-1}, ..., ε_{t-q}]
        # Simplified: [y_t, ε_{t-1}, ε_{t-2}, ..., ε_{t-q}] for AR(1)
        # Or more states if p > 1
        
        if p == 0:
            # Pure MA(q) model
            state_dim = q + 1
        elif q == 0:
            # Pure AR(p) model  
            state_dim = p
        else:
            # ARMA(p,q) - use max(p, q+1)
            state_dim = max(p, q + 1)
        
        # Build state transition matrix
        state_transition = [[0.0] * state_dim for _ in range(state_dim)]
        
        if p == 0:
            # MA(q) model: x_t = [ε_t, ε_{t-1}, ..., ε_{t-q}]
            # First row stays zero (pure innovation)
            # Shift register for errors
            for i in range(1, q):
                state_transition[i+1][i] = 1.0
        elif q == 0:
            # AR(p) model: x_t = [y_t, y_{t-1}, ..., y_{t-p+1}]
            for i in range(p):
                state_transition[0][i] = ar_coeffs[i]
            # Shift register for AR states
            for i in range(1, p):
                state_transition[i][i-1] = 1.0
        else:
            # ARMA(p,q): x_t = [y_t, ε_{t-1}, ..., ε_{t-q}] (simplified for p=1)
            # First element is AR coefficient
            state_transition[0][0] = ar_coeffs[0] if p == 1 else 0.0
            
            # Add MA coefficients
            for i in range(min(q, state_dim - 1)):
                state_transition[0][i + 1] = ma_coeffs[i]
            
            # If p > 1, need companion form
            if p > 1:
                for i in range(p):
                    state_transition[0][i] = ar_coeffs[i]
                for i in range(1, p):
                    state_transition[i][i-1] = 1.0
            else:
                # Shift register for MA errors
                for i in range(1, min(q, state_dim - 1)):
                    state_transition[i+1][i] = 1.0
        
        # Build process noise matrix
        process_noise_var = random.choice([0.1, 0.5, 1.0, 2.0])
        process_noise = [[0.0] * state_dim for _ in range(state_dim)]
        
        # Innovation enters first state
        process_noise[0][0] = process_noise_var
        
        # For MA terms, innovation also enters the error state
        if q > 0 and state_dim > 1:
            process_noise[1][1] = process_noise_var
        
        # Observation model - observe only first state (or sum for AR)
        observation_model = [[1.0] + [0.0] * (state_dim - 1)]
        
        # Observation noise
        observation_noise = random.choice([0.01, 0.05, 0.1, 0.5])
        
        return {
            'model_name': f'ARIMA({p},0,{q})',
            'state_transition': state_transition,
            'process_noise': process_noise,
            'observation_model': observation_model,
            'observation_noise': observation_noise,
        }

    model_generators = [
        (0.15, make_ets_aan),  # Increased weight for improved ETS
        (0.1, make_stochastic_dummy),
        (0.1, make_stochastic_seasonal_7),
        (0.05, make_ma_model),
        (0.05, make_ar2_model),
        (0.05, make_ucm_deterministic_trend),
        (0.1, make_ucm_random_walk_drift_ar1),
        (0.05, make_ucm_random_walk_drift_ar1_weekly_fourier),
        (0.05, make_x1_model),
        (0.1, make_hidden_state_seasonal_7),
        (0.05, make_factor_model),
        (0.05, make_ucm_deterministictrend_seasonal7),
        (0.05, make_spline_model),
        (0.05, make_locallinear_weekly_fourier),
        (0.03, make_locallinear_daily_fourier),
        (0.03, make_randomwalk_weekly_fourier),
        (0.05, make_holt_winters_damped),
        (0.05, lambda: make_seasonal_hidden_state(364)),
        (0.05, lambda: make_seasonal_hidden_state(12)),
        (0.1, lambda: make_randomly_generated(False)),
        (0.05, lambda: make_randomly_generated(True)),
        (0.05, make_dynamic_linear),
        (0.1, make_theta_equivalent),
        (0.1, make_ets_simple_exponential_smoothing),
        (0.1, make_ets_linear_trend),
        (0.1, make_ets_damped_trend),
        (0.1, make_arima_model),  # ARIMA models, favoring (1,0,4)
    ]

    weights = [item[0] for item in model_generators]
    generators = [item[1] for item in model_generators]
    selected_generator = random.choices(generators, weights=weights)[0]
    params = selected_generator()

    return finalize(params)
