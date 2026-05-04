from setuptools import find_packages, setup


setup(
    name="offline-rl-fusion",
    version="0.1.0",
    description="Offline RL and fusion-control utilities with a D4RL-style import surface.",
    install_requires=[
        "numpy>=1.22",
        "torch>=2.1",
        "h5py>=3.14",
        "tqdm>=4.62",
        "PyYAML>=6.0",
        "matplotlib>=3.4",
        "pandas>=2.1",
        "scipy>=1.11",
        "seaborn>=0.13",
        "easydict>=1.13",
        "scikit-learn>=0.24",
        "gym>=0.26",
        "gymnasium>=0.29",
        "stable-baselines3>=2.1",
        "optuna>=4.6",
        "hydra-core>=1.1",
        "omegaconf>=2.1",
        "pytorch-lightning>=1.4",
        "tabulate>=0.9",
        "uncertainty-toolbox>=0.1.1",
        "dynamics-toolbox>=1.0.0",
    ],
    packages=find_packages(
        include=[
            "offlinerlkit",
            "offlinerlkit.*",
            "envs",
            "envs.*",
            "dynamics",
            "dynamics.*",
            "rl_preparation",
            "rl_preparation.*",
            "visualization",
            "visualization.*",
        ]
    ),
    include_package_data=True,
    package_data={
        "dynamics": ["cfgs/*.yaml"],
        "offlinerlkit.configs": ["benchmark_configs.json"],
    },
    entry_points={
        "console_scripts": [
            "nf-process-raw-data=rl_preparation.process_raw_data:main",
            "nf-train-dynamics=dynamics.train_dynamics:train_ensemble",
        ]
    },
    python_requires=">=3.6",
)
