"""
Configuration for experiments.
"""

from dataclasses import dataclass


@dataclass
class ExperimentModelConfig:
    """Configuration for model hyperparameters in experiments."""
    model_name: str = "gpt2"
    lora_rank: int = 64
    hidden_dim: int = 768
    num_heads: int = 12
    head_dim: int = 64
    ttt_hidden_dim: int = 2048
    chunk_size: int = 512  # Must be <= model's max_position_embeddings (1024 for GPT-2)
    max_seq_length: int = 1024  # GPT-2 maximum
    dropout_rate: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 16
    learning_rate: float = 3e-4
    num_train_examples: int = 5000
    num_eval_examples: int = 500
    budget_limit: float = 100.0

    # PID-Lagrangian PPO parameters
    ppo_clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PID controller gains
    pid_kp: float = 0.1
    pid_ki: float = 0.01
    pid_kd: float = 0.01

    # Training loop
    num_iterations: int = 100
    rollout_length: int = 256
    ppo_epochs: int = 4

    # Logging
    log_every: int = 10
    eval_every: int = 50
    save_every: int = 100


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ExperimentModelConfig
    training: TrainingConfig

    # Experiment metadata
    experiment_name: str = "ponderttt"
    output_dir: str = "outputs"
    seed: int = 42

    # Device configuration
    use_tpu: bool = False
    mesh_shape: tuple = (8, 1)
    mesh_axes: tuple = ('batch', 'model')


def get_125m_config() -> ExperimentConfig:
    """Get configuration for 125M model (GPT-2)."""
    model = ExperimentModelConfig(
        model_name="gpt2",
        lora_rank=64,
        chunk_size=512,
    )

    training = TrainingConfig(
        batch_size=16,
        num_train_examples=5000,
        num_iterations=100,
    )

    return ExperimentConfig(
        model=model,
        training=training,
        experiment_name="ponderttt_125m",
    )


def get_350m_config() -> ExperimentConfig:
    """Get configuration for 350M model (GPT-2 Medium)."""
    model = ExperimentModelConfig(
        model_name="gpt2-medium",
        lora_rank=128,
        chunk_size=512,
    )

    training = TrainingConfig(
        batch_size=12,
        num_train_examples=10000,
        num_iterations=100,
    )

    return ExperimentConfig(
        model=model,
        training=training,
        experiment_name="ponderttt_350m",
    )


def get_1b_config() -> ExperimentConfig:
    """Get configuration for 1B model (GPT-2 Large)."""
    model = ExperimentModelConfig(
        model_name="gpt2-large",
        lora_rank=256,
        chunk_size=512,
    )

    training = TrainingConfig(
        batch_size=8,
        num_train_examples=20000,
        num_iterations=100,
    )

    return ExperimentConfig(
        model=model,
        training=training,
        experiment_name="ponderttt_1b",
    )
