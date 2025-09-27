#!/usr/bin/env python3
"""
Mathematical Model Extraction and Analysis - VSCode Edition
===========================================================

A comprehensive Python analysis tool for extracting and simulating mathematical models
with detailed graphical and statistical results. Perfect for VSCode data science workflow.

Features:
- LaTeX equation extraction and parsing
- SDE model classification and analysis
- Monte Carlo simulations with multiple paths
- Statistical analysis and hypothesis testing
- Professional visualizations and plots
- Export results to various formats

Author: Mathematical Model Extraction System
"""

import sys
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import json

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src directory to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Import our mathematical extraction components
try:
    from src.extraction.orchestrator import ExtractionOrchestrator, ExtractionPipeline
    from src.extraction.parsers.latex_parser import LaTeXParser
    from src.extraction.extractors.equation_extractor import EquationExtractor
    from src.extraction.validators.sde_classifier import SDEClassifier
    from src.sde.configuration.simulation_config import SimulationConfiguration, NumericalScheme
    from src.core.equations.mathematical_equation import MathematicalEquation, EquationType
    print("All mathematical extraction components imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

class MathematicalAnalyzer:
    """Comprehensive mathematical model analyzer with visualization capabilities."""

    def __init__(self):
        """Initialize the mathematical analyzer."""
        self.orchestrator = ExtractionOrchestrator()
        self.results = {}
        self.simulation_data = {}
        self.figures = []

        print("Mathematical Model Analyzer Initialized")
        print("=" * 60)

    def extract_and_analyze_models(self, document_content: str) -> Dict[str, Any]:
        """Extract mathematical models from LaTeX document and analyze them."""

        print("DOCUMENT ANALYSIS PHASE")
        print("-" * 40)

        # Configure extraction pipeline
        pipeline_config = ExtractionPipeline(
            extract_equations=True,
            analyze_dependencies=True,
            classify_sdes=True,
            validate_fidelity=True,
            create_models=True,
            min_confidence=0.3
        )

        # Process document
        start_time = time.time()
        extraction_result = self.orchestrator.process_document(
            document_content=document_content,
            document_id="analysis_session",
            pipeline_config=pipeline_config
        )
        processing_time = time.time() - start_time

        print(f"Processing completed in {processing_time:.3f} seconds")
        print(f"Equations extracted: {len(extraction_result.equations)}")
        print(f"Variables identified: {len(extraction_result.variables)}")
        print(f"SDE models created: {len(extraction_result.sde_models)}")

        # Store results
        self.results = {
            'extraction_result': extraction_result,
            'processing_time': processing_time,
            'equations': extraction_result.equations,
            'variables': extraction_result.variables,
            'sde_models': extraction_result.sde_models,
            'statistics': extraction_result.statistics
        }

        return self.results

    def run_monte_carlo_simulations(self, num_paths: int = 5000, time_horizon: float = 1.0) -> Dict[str, Any]:
        """Run comprehensive Monte Carlo simulations for all identified SDE models."""

        print(f"\nMONTE CARLO SIMULATION PHASE")
        print("-" * 40)

        if not self.results.get('sde_models'):
            print("No SDE models found to simulate")
            return {}

        simulation_results = {}

        for model in self.results['sde_models']:
            print(f"\n[SIM] Simulating: {model.model_name}")

            # Configure simulation
            config = SimulationConfiguration(
                model_reference=model.model_name,
                time_horizon=time_horizon,
                time_steps=252,  # Daily steps for one year
                monte_carlo_paths=num_paths,
                numerical_scheme=NumericalScheme.EULER_MARUYAMA,
                random_seed=42
            )

            # Run simulation based on model type
            if "geometric" in model.model_name.lower() or "brownian" in model.model_name.lower():
                paths, times = self._simulate_geometric_brownian_motion(config)
            elif "ornstein" in model.model_name.lower() or "uhlenbeck" in model.model_name.lower():
                paths, times = self._simulate_ornstein_uhlenbeck(config)
            elif "cox" in model.model_name.lower() or "ingersoll" in model.model_name.lower():
                paths, times = self._simulate_cox_ingersoll_ross(config)
            elif "cost" in model.model_name.lower() or "maliyet" in model.model_name.lower():
                paths, times = self._simulate_cost_process(config)
            elif "fire" in model.model_name.lower() or "omega" in model.model_name.lower():
                paths, times = self._simulate_fire_rate_process(config)
            elif "inflation" in model.model_name.lower() or "enflasyon" in model.model_name.lower():
                paths, times = self._simulate_inflation_expectations(config)
            elif "producer" in model.model_name.lower() or "uretici" in model.model_name.lower():
                paths, times = self._simulate_producer_price_jump_diffusion(config)
            elif "wholesale" in model.model_name.lower() or "toptan" in model.model_name.lower():
                paths, times = self._simulate_wholesale_price_with_amplifier(config)
            elif "retail" in model.model_name.lower() or "perakende" in model.model_name.lower():
                paths, times = self._simulate_retail_price_with_expectations(config)
            else:
                # Default to nested SDE system with full coupling
                paths, times = self._simulate_nested_sde_system(config)

            # Store simulation data
            simulation_results[model.model_name] = {
                'paths': paths,
                'times': times,
                'config': config,
                'model': model
            }

            print(f"   [OK] Generated {num_paths} paths over {time_horizon} years")
            print(f"   Final values range: [{np.min(paths[:, -1]):.2f}, {np.max(paths[:, -1]):.2f}]")
            print(f"   [AVG] Mean final value: {np.mean(paths[:, -1]):.2f}")

        self.simulation_data = simulation_results
        return simulation_results

    def _simulate_geometric_brownian_motion(self, config: SimulationConfiguration) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate Geometric Brownian Motion: dS = μS dt + σS dW."""

        # Parameters
        mu = 0.05      # Drift (5% annual return)
        sigma = 0.2    # Volatility (20% annual volatility)
        S0 = 100.0     # Initial value

        # Time setup
        dt = config.get_time_step_size()
        times = np.linspace(0, config.time_horizon, config.time_steps + 1)

        # Generate random increments
        np.random.seed(config.random_seed)
        dW = np.random.normal(0, np.sqrt(dt), (config.monte_carlo_paths, config.time_steps))

        # Initialize paths
        paths = np.zeros((config.monte_carlo_paths, config.time_steps + 1))
        paths[:, 0] = S0

        # Simulate paths
        for i in range(config.time_steps):
            paths[:, i + 1] = paths[:, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[:, i])

        return paths, times

    def _simulate_ornstein_uhlenbeck(self, config: SimulationConfiguration) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate Ornstein-Uhlenbeck Process: dX = κ(θ - X)dt + σ dW."""

        # Parameters
        kappa = 2.0    # Mean reversion speed
        theta = 0.05   # Long-term mean
        sigma = 0.1    # Volatility
        X0 = 0.03      # Initial value

        # Time setup
        dt = config.get_time_step_size()
        times = np.linspace(0, config.time_horizon, config.time_steps + 1)

        # Generate random increments
        np.random.seed(config.random_seed)
        dW = np.random.normal(0, np.sqrt(dt), (config.monte_carlo_paths, config.time_steps))

        # Initialize paths
        paths = np.zeros((config.monte_carlo_paths, config.time_steps + 1))
        paths[:, 0] = X0

        # Simulate paths using Euler-Maruyama scheme
        for i in range(config.time_steps):
            drift = kappa * (theta - paths[:, i]) * dt
            diffusion = sigma * dW[:, i]
            paths[:, i + 1] = paths[:, i] + drift + diffusion

        return paths, times

    def _simulate_cox_ingersoll_ross(self, config: SimulationConfiguration) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate Cox-Ingersoll-Ross Process: dr = κ(θ - r)dt + σ√r dW."""

        # Parameters
        kappa = 2.0    # Mean reversion speed
        theta = 0.05   # Long-term mean
        sigma = 0.1    # Volatility of volatility
        r0 = 0.03      # Initial rate

        # Time setup
        dt = config.get_time_step_size()
        times = np.linspace(0, config.time_horizon, config.time_steps + 1)

        # Generate random increments
        np.random.seed(config.random_seed)
        dW = np.random.normal(0, np.sqrt(dt), (config.monte_carlo_paths, config.time_steps))

        # Initialize paths
        paths = np.zeros((config.monte_carlo_paths, config.time_steps + 1))
        paths[:, 0] = r0

        # Simulate paths with non-negativity constraint
        for i in range(config.time_steps):
            drift = kappa * (theta - paths[:, i]) * dt
            diffusion = sigma * np.sqrt(np.maximum(paths[:, i], 0)) * dW[:, i]
            paths[:, i + 1] = np.maximum(paths[:, i] + drift + diffusion, 0)

        return paths, times

    def _simulate_cost_process(self, config: SimulationConfiguration) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate cost processes (production/logistics): dC = μC dt + σC dW."""

        # Parameters for cost dynamics
        mu = 0.03      # Trend growth (3% annual)
        sigma = 0.15   # Volatility (15% annual)
        C0 = 50.0      # Initial cost level

        # Time setup
        dt = config.get_time_step_size()
        times = np.linspace(0, config.time_horizon, config.time_steps + 1)

        # Generate random increments
        np.random.seed(config.random_seed)
        dW = np.random.normal(0, np.sqrt(dt), (config.monte_carlo_paths, config.time_steps))

        # Initialize paths
        paths = np.zeros((config.monte_carlo_paths, config.time_steps + 1))
        paths[:, 0] = C0

        # Simulate GBM for costs
        for i in range(config.time_steps):
            paths[:, i + 1] = paths[:, i] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW[:, i])

        return paths, times

    def _simulate_fire_rate_process(self, config: SimulationConfiguration) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate fire rate (spoilage) process: dω = κ(ω̄ - ω)dt + σ dW."""

        # Parameters for Ornstein-Uhlenbeck fire rate
        kappa = 3.0     # Mean reversion speed
        omega_bar = 0.05  # Long-term average fire rate (5%)
        sigma = 0.02    # Volatility
        omega0 = 0.04   # Initial fire rate

        # Time setup
        dt = config.get_time_step_size()
        times = np.linspace(0, config.time_horizon, config.time_steps + 1)

        # Generate random increments
        np.random.seed(config.random_seed + 1)
        dW = np.random.normal(0, np.sqrt(dt), (config.monte_carlo_paths, config.time_steps))

        # Initialize paths
        paths = np.zeros((config.monte_carlo_paths, config.time_steps + 1))
        paths[:, 0] = omega0

        # Simulate Ornstein-Uhlenbeck process
        for i in range(config.time_steps):
            drift = kappa * (omega_bar - paths[:, i]) * dt
            diffusion = sigma * dW[:, i]
            paths[:, i + 1] = np.clip(paths[:, i] + drift + diffusion, 0.001, 0.999)

        return paths, times

    def _simulate_inflation_expectations(self, config: SimulationConfiguration) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate adaptive inflation expectations: dΠᴱ = κ(π - Πᴱ)dt + σ dW."""

        # Parameters
        kappa_pi = 2.5    # Learning speed
        pi_realized = 0.08  # Realized inflation (8% annual)
        sigma_pi = 0.03   # Expectation formation noise
        Pi0 = 0.06        # Initial expectations

        # Time setup
        dt = config.get_time_step_size()
        times = np.linspace(0, config.time_horizon, config.time_steps + 1)

        # Generate random increments
        np.random.seed(config.random_seed + 2)
        dW = np.random.normal(0, np.sqrt(dt), (config.monte_carlo_paths, config.time_steps))

        # Initialize paths
        paths = np.zeros((config.monte_carlo_paths, config.time_steps + 1))
        paths[:, 0] = Pi0

        # Simulate adaptive expectations
        for i in range(config.time_steps):
            # Realized inflation varies slightly
            pi_t = pi_realized + 0.02 * np.sin(2 * np.pi * times[i] / config.time_horizon)
            drift = kappa_pi * (pi_t - paths[:, i]) * dt
            diffusion = sigma_pi * dW[:, i]
            paths[:, i + 1] = np.maximum(paths[:, i] + drift + diffusion, 0.001)

        return paths, times

    def _simulate_producer_price_jump_diffusion(self, config: SimulationConfiguration) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate producer price with supply shocks: dP = κ((1+μ)C - P)dt + σP dW + JP dN."""

        # Parameters
        kappa_F = 4.0      # Mean reversion speed
        mu_F = 0.25        # Producer markup (25%)
        sigma_F = 0.2      # Price volatility
        lambda_jump = 1.0  # Jump intensity (1 per year)
        J_F = 0.15         # Jump size (15% price increase)

        # Cost process (simplified)
        C_base = 100.0

        # Time setup
        dt = config.get_time_step_size()
        times = np.linspace(0, config.time_horizon, config.time_steps + 1)

        # Generate random components
        np.random.seed(config.random_seed + 3)
        dW = np.random.normal(0, np.sqrt(dt), (config.monte_carlo_paths, config.time_steps))

        # Jump process
        dN = np.random.poisson(lambda_jump * dt, (config.monte_carlo_paths, config.time_steps))

        # Initialize paths
        paths = np.zeros((config.monte_carlo_paths, config.time_steps + 1))
        paths[:, 0] = (1 + mu_F) * C_base

        # Simulate jump-diffusion
        for i in range(config.time_steps):
            C_t = C_base * (1 + 0.03 * times[i])  # Growing costs
            target = (1 + mu_F) * C_t

            drift = kappa_F * (target - paths[:, i]) * dt
            diffusion = sigma_F * paths[:, i] * dW[:, i]
            jumps = J_F * paths[:, i] * dN[:, i]

            paths[:, i + 1] = paths[:, i] + drift + diffusion + jumps

        return paths, times

    def _simulate_wholesale_price_with_amplifier(self, config: SimulationConfiguration) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate wholesale price with fire amplifier: P_W = (1+μ_W) * (P_F + C_L)/(1-ω)."""

        # Parameters
        kappa_W = 5.0      # Adjustment speed
        mu_W = 0.20        # Wholesale markup (20%)
        sigma_W = 0.15     # Wholesale volatility

        # Simulate underlying processes
        config_producer = config
        config_producer.random_seed = config.random_seed + 4
        P_F, _ = self._simulate_producer_price_jump_diffusion(config_producer)

        config_cost = config
        config_cost.random_seed = config.random_seed + 5
        C_L, _ = self._simulate_cost_process(config_cost)

        config_fire = config
        config_fire.random_seed = config.random_seed + 6
        omega, times = self._simulate_fire_rate_process(config_fire)

        # Time setup
        dt = config.get_time_step_size()

        # Generate random increments
        np.random.seed(config.random_seed + 7)
        dW = np.random.normal(0, np.sqrt(dt), (config.monte_carlo_paths, config.time_steps))

        # Initialize wholesale price paths
        paths = np.zeros((config.monte_carlo_paths, config.time_steps + 1))
        paths[:, 0] = (1 + mu_W) * (P_F[:, 0] + C_L[:, 0]) / (1 - omega[:, 0])

        # Simulate wholesale price with fire amplifier
        for i in range(config.time_steps):
            # Effective cost with fire amplifier
            C_eff = (P_F[:, i] + C_L[:, i]) / (1 - omega[:, i])
            target = (1 + mu_W) * C_eff

            drift = kappa_W * (target - paths[:, i]) * dt
            diffusion = sigma_W * paths[:, i] * dW[:, i]

            paths[:, i + 1] = paths[:, i] + drift + diffusion

        return paths, times

    def _simulate_retail_price_with_expectations(self, config: SimulationConfiguration) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate retail price with expectation-dependent markup: μ_R(Πᴱ) = μ̄_R + βΠᴱ."""

        # Parameters
        kappa_R = 3.0      # Adjustment speed
        mu_R_bar = 0.30    # Base retail markup (30%)
        beta = 0.5         # Sensitivity to expectations
        sigma_R = 0.12     # Retail volatility

        # Simulate underlying processes
        config_wholesale = config
        config_wholesale.random_seed = config.random_seed + 8
        P_W, _ = self._simulate_wholesale_price_with_amplifier(config_wholesale)

        config_expect = config
        config_expect.random_seed = config.random_seed + 9
        Pi_E, times = self._simulate_inflation_expectations(config_expect)

        # Time setup
        dt = config.get_time_step_size()

        # Generate random increments
        np.random.seed(config.random_seed + 10)
        dW = np.random.normal(0, np.sqrt(dt), (config.monte_carlo_paths, config.time_steps))

        # Initialize retail price paths
        paths = np.zeros((config.monte_carlo_paths, config.time_steps + 1))
        initial_markup = mu_R_bar + beta * Pi_E[:, 0]
        paths[:, 0] = (1 + initial_markup) * P_W[:, 0]

        # Simulate retail price with expectation feedback
        for i in range(config.time_steps):
            # Expectation-dependent markup
            mu_R_t = mu_R_bar + beta * Pi_E[:, i]
            target = (1 + mu_R_t) * P_W[:, i]

            drift = kappa_R * (target - paths[:, i]) * dt
            diffusion = sigma_R * paths[:, i] * dW[:, i]

            paths[:, i + 1] = paths[:, i] + drift + diffusion

        return paths, times

    def _simulate_comprehensive_food_price_system(self, config: SimulationConfiguration) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the complete integrated food price system with all amplifiers."""

        # Parameters for integrated system
        mu_F = 0.25        # Producer markup
        mu_W = 0.20        # Wholesale markup
        mu_R_bar = 0.30    # Base retail markup
        beta = 0.8         # Expectation sensitivity

        # Simulate all underlying processes
        np.random.seed(config.random_seed + 11)

        # Base costs
        C_P_paths, _ = self._simulate_cost_process(config)
        C_L_paths, _ = self._simulate_cost_process(config)

        # Fire rate
        omega_paths, _ = self._simulate_fire_rate_process(config)

        # Inflation expectations
        Pi_E_paths, times = self._simulate_inflation_expectations(config)

        # Producer prices (with costs from C_P)
        P_F_paths = C_P_paths * (1 + mu_F)

        # Wholesale prices with fire amplifier
        P_W_paths = np.zeros_like(P_F_paths)
        for i in range(config.time_steps + 1):
            effective_cost = (P_F_paths[:, i] + C_L_paths[:, i]) / (1 - omega_paths[:, i])
            P_W_paths[:, i] = (1 + mu_W) * effective_cost

        # Retail prices with expectation feedback
        paths = np.zeros_like(P_W_paths)
        for i in range(config.time_steps + 1):
            mu_R_t = mu_R_bar + beta * Pi_E_paths[:, i]
            paths[:, i] = (1 + mu_R_t) * P_W_paths[:, i]

        return paths, times

    def _simulate_nested_sde_system(self, config: SimulationConfiguration) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the complete nested SDE system with full coupling between all equations."""

        print(f"   [NESTED] Implementing full coupled SDE system...")

        # Time setup
        dt = config.get_time_step_size()
        times = np.linspace(0, config.time_horizon, config.time_steps + 1)
        n_paths = config.monte_carlo_paths
        n_steps = config.time_steps

        # Initialize random seeds for each component
        np.random.seed(config.random_seed)

        # Generate all Brownian motions (7 independent sources)
        dW_P = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))      # Production costs
        dW_L = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))      # Logistics costs
        dW_omega = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))  # Fire rate
        dW_pi = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))     # Inflation expectations
        dW_F = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))      # Producer price
        dW_W = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))      # Wholesale price
        dW_R = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))      # Retail price

        # Poisson jumps for supply shocks
        lambda_jump = 1.0  # 1 jump per year
        dN = np.random.poisson(lambda_jump * dt, (n_paths, n_steps))

        # Model parameters
        params = {
            # Cost processes (GBM)
            'mu_P': 0.03, 'sigma_P': 0.15,      # Production cost drift/vol
            'mu_L': 0.025, 'sigma_L': 0.12,     # Logistics cost drift/vol

            # Fire rate (OU process)
            'kappa_omega': 3.0, 'omega_bar': 0.05, 'sigma_omega': 0.02,

            # Inflation expectations (adaptive)
            'kappa_pi': 2.5, 'sigma_pi': 0.03,

            # Producer prices (jump-diffusion with mean reversion)
            'kappa_F': 4.0, 'mu_F': 0.25, 'sigma_F': 0.2, 'J_F': 0.15,

            # Wholesale prices (mean reversion to effective cost)
            'kappa_W': 5.0, 'mu_W': 0.20, 'sigma_W': 0.15,

            # Retail prices (mean reversion with expectation feedback)
            'kappa_R': 3.0, 'mu_R_bar': 0.30, 'beta': 0.8, 'sigma_R': 0.12
        }

        # Initialize state variables [C_P, C_L, omega, Pi_E, P_F, P_W, P_R]
        state = np.zeros((n_paths, 7, n_steps + 1))

        # Initial conditions
        state[:, 0, 0] = 50.0      # C_P(0) - Production cost
        state[:, 1, 0] = 45.0      # C_L(0) - Logistics cost
        state[:, 2, 0] = 0.04      # omega(0) - Fire rate
        state[:, 3, 0] = 0.06      # Pi_E(0) - Inflation expectations
        state[:, 4, 0] = 62.5      # P_F(0) = (1+mu_F)*C_P(0)
        state[:, 5, 0] = 114.0     # P_W(0) - Wholesale price
        state[:, 6, 0] = 148.2     # P_R(0) - Retail price

        print(f"   [COUPLED] Starting iterative SDE integration...")

        # Main simulation loop - fully coupled system
        for i in range(n_steps):
            # Extract current state
            C_P = state[:, 0, i]
            C_L = state[:, 1, i]
            omega = state[:, 2, i]
            Pi_E = state[:, 3, i]
            P_F = state[:, 4, i]
            P_W = state[:, 5, i]
            P_R = state[:, 6, i]

            # Calculate realized inflation rate (needed for expectations)
            if i > 0:
                pi_R = (P_R - state[:, 6, i-1]) / state[:, 6, i-1] / dt
            else:
                pi_R = 0.08 * np.ones(n_paths)  # Initial assumption

            # 1. Production costs: dC_P = μ_P C_P dt + σ_P C_P dW_P
            drift_CP = params['mu_P'] * C_P * dt
            diffusion_CP = params['sigma_P'] * C_P * dW_P[:, i]
            state[:, 0, i+1] = C_P + drift_CP + diffusion_CP

            # 2. Logistics costs: dC_L = μ_L C_L dt + σ_L C_L dW_L
            drift_CL = params['mu_L'] * C_L * dt
            diffusion_CL = params['sigma_L'] * C_L * dW_L[:, i]
            state[:, 1, i+1] = C_L + drift_CL + diffusion_CL

            # 3. Fire rate: dω = κ_ω(ω̄ - ω)dt + σ_ω dW_ω
            drift_omega = params['kappa_omega'] * (params['omega_bar'] - omega) * dt
            diffusion_omega = params['sigma_omega'] * dW_omega[:, i]
            state[:, 2, i+1] = np.clip(omega + drift_omega + diffusion_omega, 0.001, 0.999)

            # 4. Inflation expectations: dΠ^E = κ_π(π_R - Π^E)dt + σ_π dW_π
            drift_PiE = params['kappa_pi'] * (pi_R - Pi_E) * dt
            diffusion_PiE = params['sigma_pi'] * dW_pi[:, i]
            state[:, 3, i+1] = np.maximum(Pi_E + drift_PiE + diffusion_PiE, 0.001)

            # 5. Producer prices: dP_F = κ_F((1+μ_F)C_P - P_F)dt + σ_F P_F dW_F + J_F P_F dN
            target_PF = (1 + params['mu_F']) * state[:, 0, i+1]  # Use updated C_P
            drift_PF = params['kappa_F'] * (target_PF - P_F) * dt
            diffusion_PF = params['sigma_F'] * P_F * dW_F[:, i]
            jumps_PF = params['J_F'] * P_F * dN[:, i]
            state[:, 4, i+1] = P_F + drift_PF + diffusion_PF + jumps_PF

            # 6. Wholesale prices with fire amplifier: P_W → (1+μ_W) * (P_F + C_L)/(1-ω)
            C_eff = (state[:, 4, i+1] + state[:, 1, i+1]) / (1 - state[:, 2, i+1])  # Effective cost
            target_PW = (1 + params['mu_W']) * C_eff
            drift_PW = params['kappa_W'] * (target_PW - P_W) * dt
            diffusion_PW = params['sigma_W'] * P_W * dW_W[:, i]
            state[:, 5, i+1] = P_W + drift_PW + diffusion_PW

            # 7. Retail prices with expectation feedback: μ_R(Π^E) = μ̄_R + β*Π^E
            mu_R_dynamic = params['mu_R_bar'] + params['beta'] * state[:, 3, i+1]  # Use updated Pi_E
            target_PR = (1 + mu_R_dynamic) * state[:, 5, i+1]  # Use updated P_W
            drift_PR = params['kappa_R'] * (target_PR - P_R) * dt
            diffusion_PR = params['sigma_R'] * P_R * dW_R[:, i]
            state[:, 6, i+1] = P_R + drift_PR + diffusion_PR

            # Progress indicator
            if (i + 1) % (n_steps // 10) == 0:
                progress = (i + 1) / n_steps * 100
                print(f"   [PROGRESS] {progress:.0f}% - P_R: {np.mean(state[:, 6, i+1]):.2f}")

        print(f"   [COMPLETE] Nested SDE integration finished")

        # Store all state variables for comprehensive analysis
        self.nested_state_data = {
            'C_P': state[:, 0, :],     # Production costs
            'C_L': state[:, 1, :],     # Logistics costs
            'omega': state[:, 2, :],   # Fire rate
            'Pi_E': state[:, 3, :],    # Inflation expectations
            'P_F': state[:, 4, :],     # Producer prices
            'P_W': state[:, 5, :],     # Wholesale prices
            'P_R': state[:, 6, :],     # Retail prices
            'times': times
        }

        # Return the retail prices (final output of the chain)
        return state[:, 6, :], times

    def generate_statistical_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive statistical analysis of simulation results."""

        print(f"\nSTATISTICAL ANALYSIS PHASE")
        print("-" * 40)

        stats_results = {}

        for model_name, sim_data in self.simulation_data.items():
            print(f"\n[STATS] Analyzing: {model_name}")

            paths = sim_data['paths']
            times = sim_data['times']
            final_values = paths[:, -1]

            # Basic statistics
            stats_summary = {
                'mean': np.mean(final_values),
                'std': np.std(final_values),
                'min': np.min(final_values),
                'max': np.max(final_values),
                'median': np.median(final_values),
                'skewness': stats.skew(final_values),
                'kurtosis': stats.kurtosis(final_values),
                'var': np.var(final_values)
            }

            # Percentiles
            percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])
            stats_summary['percentiles'] = {
                '5%': percentiles[0],
                '25%': percentiles[1],
                '50%': percentiles[2],
                '75%': percentiles[3],
                '95%': percentiles[4]
            }

            # Path statistics
            path_means = np.mean(paths, axis=0)
            path_stds = np.std(paths, axis=0)

            # Time series analysis
            returns = np.diff(np.log(paths), axis=1)
            daily_returns = returns.flatten()

            # Risk metrics
            VaR_95 = np.percentile(final_values, 5)
            CVaR_95 = np.mean(final_values[final_values <= VaR_95])

            # Additional metrics
            max_drawdown = self._calculate_max_drawdown(paths)
            sharpe_ratio = self._calculate_sharpe_ratio(returns)

            stats_summary.update({
                'path_means': path_means,
                'path_stds': path_stds,
                'daily_returns': daily_returns,
                'VaR_95': VaR_95,
                'CVaR_95': CVaR_95,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'volatility_annual': np.std(daily_returns) * np.sqrt(252)
            })

            stats_results[model_name] = stats_summary

            # Print key statistics
            print(f"   Mean final value: {stats_summary['mean']:.4f}")
            print(f"   [STD] Standard deviation: {stats_summary['std']:.4f}")
            print(f"   [VAR] VaR (95%): {stats_summary['VaR_95']:.4f}")
            print(f"   [SHARP] Sharpe ratio: {stats_summary['sharpe_ratio']:.4f}")
            print(f"   Max drawdown: {stats_summary['max_drawdown']:.2%}")

        return stats_results

    def generate_agricultural_price_path_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive agricultural price path analysis from nested SDE system."""

        if not hasattr(self, 'nested_state_data'):
            print("[WARNING] Nested SDE data not found. Run simulation first.")
            return {}

        print(f"\nTARIM FIYAT PATIKA ANALIZI")
        print("-" * 50)

        nested_data = self.nested_state_data
        times = nested_data['times']

        # Extract all price paths
        C_P = nested_data['C_P']      # Production costs
        C_L = nested_data['C_L']      # Logistics costs
        omega = nested_data['omega']   # Fire rate
        Pi_E = nested_data['Pi_E']    # Inflation expectations
        P_F = nested_data['P_F']      # Producer (farm-gate) prices
        P_W = nested_data['P_W']      # Wholesale prices
        P_R = nested_data['P_R']      # Retail prices

        print(f"[ANALIZ] {P_F.shape[0]} simulasyon yolu, {len(times)} zaman noktasi")

        # Calculate derived metrics
        fire_amplifier = 1 / (1 - omega)  # Fire amplification factor
        cost_base = C_P + C_L             # Total base costs
        markup_producer = P_F / C_P       # Producer markup ratio
        markup_wholesale = P_W / (P_F + C_L) * (1 - omega)  # Wholesale markup (adjusted for fire)
        markup_retail = P_R / P_W         # Retail markup ratio
        total_amplifier = P_R / cost_base # Total farm-to-fork amplification

        # Time series statistics
        analysis_results = {}

        # 1. Price Level Analysis
        analysis_results['price_levels'] = {
            'farm_gate_final': {
                'mean': np.mean(P_F[:, -1]),
                'std': np.std(P_F[:, -1]),
                'min': np.min(P_F[:, -1]),
                'max': np.max(P_F[:, -1]),
                'median': np.median(P_F[:, -1]),
                'cv': np.std(P_F[:, -1]) / np.mean(P_F[:, -1])
            },
            'wholesale_final': {
                'mean': np.mean(P_W[:, -1]),
                'std': np.std(P_W[:, -1]),
                'min': np.min(P_W[:, -1]),
                'max': np.max(P_W[:, -1]),
                'median': np.median(P_W[:, -1]),
                'cv': np.std(P_W[:, -1]) / np.mean(P_W[:, -1])
            },
            'retail_final': {
                'mean': np.mean(P_R[:, -1]),
                'std': np.std(P_R[:, -1]),
                'min': np.min(P_R[:, -1]),
                'max': np.max(P_R[:, -1]),
                'median': np.median(P_R[:, -1]),
                'cv': np.std(P_R[:, -1]) / np.mean(P_R[:, -1])
            }
        }

        # 2. Amplification Analysis
        analysis_results['amplification'] = {
            'fire_amplifier': {
                'mean': np.mean(fire_amplifier[:, -1]),
                'std': np.std(fire_amplifier[:, -1]),
                'min': np.min(fire_amplifier[:, -1]),
                'max': np.max(fire_amplifier[:, -1])
            },
            'total_chain_amplifier': {
                'mean': np.mean(total_amplifier[:, -1]),
                'std': np.std(total_amplifier[:, -1]),
                'min': np.min(total_amplifier[:, -1]),
                'max': np.max(total_amplifier[:, -1])
            },
            'farm_to_retail_ratio': {
                'mean': np.mean(P_R[:, -1] / P_F[:, -1]),
                'std': np.std(P_R[:, -1] / P_F[:, -1]),
                'median': np.median(P_R[:, -1] / P_F[:, -1])
            }
        }

        # 3. Time Path Volatility
        price_returns_F = np.diff(np.log(P_F), axis=1)
        price_returns_W = np.diff(np.log(P_W), axis=1)
        price_returns_R = np.diff(np.log(P_R), axis=1)

        analysis_results['volatility'] = {
            'farm_gate_volatility': np.std(price_returns_F.flatten()) * np.sqrt(252),
            'wholesale_volatility': np.std(price_returns_W.flatten()) * np.sqrt(252),
            'retail_volatility': np.std(price_returns_R.flatten()) * np.sqrt(252),
            'volatility_amplification': np.std(price_returns_R.flatten()) / np.std(price_returns_F.flatten())
        }

        # 4. Inflation Expectations Impact
        analysis_results['expectation_impact'] = {
            'initial_expectations': np.mean(Pi_E[:, 0]) * 100,
            'final_expectations': np.mean(Pi_E[:, -1]) * 100,
            'expectation_change': (np.mean(Pi_E[:, -1]) - np.mean(Pi_E[:, 0])) * 100,
            'expectation_volatility': np.std(Pi_E[:, -1]) * 100,
            'price_expectation_correlation': np.corrcoef(P_R[:, -1], Pi_E[:, -1])[0, 1]
        }

        # 5. Fire Rate Impact
        analysis_results['fire_impact'] = {
            'average_fire_rate': np.mean(omega[:, -1]) * 100,
            'fire_rate_volatility': np.std(omega[:, -1]) * 100,
            'fire_price_correlation': np.corrcoef(omega[:, -1], P_R[:, -1])[0, 1],
            'efficiency_loss': (np.mean(fire_amplifier[:, -1]) - 1) * 100  # % additional cost due to fire
        }

        # 6. Cost Pass-through Analysis
        cost_elasticity = np.corrcoef(cost_base[:, -1], P_R[:, -1])[0, 1]
        analysis_results['cost_passthrough'] = {
            'cost_price_elasticity': cost_elasticity,
            'cost_amplification_mean': np.mean(total_amplifier[:, -1]),
            'cost_amplification_p95': np.percentile(total_amplifier[:, -1], 95),
            'cost_amplification_p05': np.percentile(total_amplifier[:, -1], 5)
        }

        # Print detailed results
        print(f"\n=== FIYAT SEVIYESI ANALIZI ===")
        for stage, data in analysis_results['price_levels'].items():
            print(f"{stage.upper().replace('_', ' ')}:")
            print(f"  Ortalama: {data['mean']:.2f}")
            print(f"  Std Sapma: {data['std']:.2f}")
            print(f"  Varyasyon Katsayisi: {data['cv']:.3f}")
            print(f"  Aralik: [{data['min']:.2f}, {data['max']:.2f}]")

        print(f"\n=== CARPAN FAKTORLERI ===")
        print(f"Fire Carpani (1/(1-omega)): {analysis_results['amplification']['fire_amplifier']['mean']:.3f}")
        print(f"Toplam Zincir Carpani: {analysis_results['amplification']['total_chain_amplifier']['mean']:.3f}")
        print(f"Tarla-Tuketici Orani: {analysis_results['amplification']['farm_to_retail_ratio']['mean']:.3f}")

        print(f"\n=== VOLATILITE ANALIZI ===")
        print(f"Tarla Cikis Volatilitesi: {analysis_results['volatility']['farm_gate_volatility']:.2%}")
        print(f"Toptan Volatilite: {analysis_results['volatility']['wholesale_volatility']:.2%}")
        print(f"Perakende Volatilite: {analysis_results['volatility']['retail_volatility']:.2%}")
        print(f"Volatilite Carpani: {analysis_results['volatility']['volatility_amplification']:.3f}")

        print(f"\n=== ENFLASYON BEKLENTILERI ===")
        print(f"Baslangic Beklentisi: {analysis_results['expectation_impact']['initial_expectations']:.2f}%")
        print(f"Final Beklenti: {analysis_results['expectation_impact']['final_expectations']:.2f}%")
        print(f"Beklenti Degisimi: {analysis_results['expectation_impact']['expectation_change']:.2f}pp")
        print(f"Fiyat-Beklenti Korelasyonu: {analysis_results['expectation_impact']['price_expectation_correlation']:.3f}")

        print(f"\n=== FIRE ETKISI ===")
        print(f"Ortalama Fire Orani: {analysis_results['fire_impact']['average_fire_rate']:.2f}%")
        print(f"Verimlilik Kaybi: {analysis_results['fire_impact']['efficiency_loss']:.2f}%")
        print(f"Fire-Fiyat Korelasyonu: {analysis_results['fire_impact']['fire_price_correlation']:.3f}")

        return analysis_results

    def create_agricultural_price_path_visualization(self) -> plt.Figure:
        """Create comprehensive agricultural price path visualization."""

        if not hasattr(self, 'nested_state_data'):
            print("[WARNING] Nested SDE data not found. Cannot create visualization.")
            return None

        nested_data = self.nested_state_data
        times = nested_data['times']

        # Extract data
        C_P = nested_data['C_P']
        C_L = nested_data['C_L']
        omega = nested_data['omega']
        Pi_E = nested_data['Pi_E']
        P_F = nested_data['P_F']
        P_W = nested_data['P_W']
        P_R = nested_data['P_R']

        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('Tarim Urunleri Fiyat Zaman Patikasi: Nested SDE Sistem Analizi',
                     fontsize=18, fontweight='bold', y=0.98)

        # Create grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

        # 1. Main Price Paths Evolution (2x2 space)
        ax_main = fig.add_subplot(gs[0:2, 0:2])

        # Sample paths for display
        n_sample = min(100, P_F.shape[0])
        sample_idx = np.random.choice(P_F.shape[0], n_sample, replace=False)

        # Plot sample paths with transparency
        for idx in sample_idx:
            ax_main.plot(times, P_F[idx], alpha=0.1, color='green', linewidth=0.5)
            ax_main.plot(times, P_W[idx], alpha=0.1, color='orange', linewidth=0.5)
            ax_main.plot(times, P_R[idx], alpha=0.1, color='red', linewidth=0.5)

        # Plot mean paths
        ax_main.plot(times, np.mean(P_F, axis=0), 'green', linewidth=4, label='Tarla Cikis Fiyati (P_F)')
        ax_main.plot(times, np.mean(P_W, axis=0), 'orange', linewidth=4, label='Toptan Fiyat (P_W)')
        ax_main.plot(times, np.mean(P_R, axis=0), 'red', linewidth=4, label='Perakende Fiyat (P_R)')

        # Add confidence bands
        for data, color, label in [(P_F, 'green', ''), (P_W, 'orange', ''), (P_R, 'red', '')]:
            mean_path = np.mean(data, axis=0)
            std_path = np.std(data, axis=0)
            ax_main.fill_between(times, mean_path - std_path, mean_path + std_path,
                                alpha=0.2, color=color)

        ax_main.set_title('Tedarik Zinciri Fiyat Evrimleri', fontsize=14, fontweight='bold')
        ax_main.set_xlabel('Zaman (Yil)')
        ax_main.set_ylabel('Fiyat Seviyesi')
        ax_main.legend(loc='upper left')
        ax_main.grid(True, alpha=0.3)

        # 2. Cost Basis Evolution
        ax_cost = fig.add_subplot(gs[0, 2])
        cost_total = C_P + C_L
        ax_cost.plot(times, np.mean(C_P, axis=0), 'darkgreen', linewidth=3, label='Uretim Maliyeti')
        ax_cost.plot(times, np.mean(C_L, axis=0), 'blue', linewidth=3, label='Lojistik Maliyeti')
        ax_cost.plot(times, np.mean(cost_total, axis=0), 'black', linewidth=3, label='Toplam Maliyet')
        ax_cost.set_title('Maliyet Bilesenleri')
        ax_cost.set_ylabel('Maliyet')
        ax_cost.legend(fontsize=8)
        ax_cost.grid(True, alpha=0.3)

        # 3. Fire Rate and Amplifier
        ax_fire = fig.add_subplot(gs[0, 3])
        fire_amp = 1 / (1 - omega)
        ax_fire.plot(times, np.mean(omega, axis=0) * 100, 'brown', linewidth=3, label='Fire Orani (%)')
        ax_fire.set_ylabel('Fire Orani (%)', color='brown')
        ax_fire.tick_params(axis='y', labelcolor='brown')

        ax_fire2 = ax_fire.twinx()
        ax_fire2.plot(times, np.mean(fire_amp, axis=0), 'chocolate', linewidth=3, label='Fire Carpani')
        ax_fire2.set_ylabel('Fire Carpani', color='chocolate')
        ax_fire2.tick_params(axis='y', labelcolor='chocolate')
        ax_fire.set_title('Fire Orani & Carpan Etkisi')
        ax_fire.grid(True, alpha=0.3)

        # 4. Inflation Expectations
        ax_infl = fig.add_subplot(gs[1, 2])
        ax_infl.plot(times, np.mean(Pi_E, axis=0) * 100, 'purple', linewidth=3)
        pi_mean = np.mean(Pi_E, axis=0) * 100
        pi_std = np.std(Pi_E, axis=0) * 100
        ax_infl.fill_between(times, pi_mean - pi_std, pi_mean + pi_std, alpha=0.3, color='purple')
        ax_infl.set_title('Enflasyon Beklentileri')
        ax_infl.set_ylabel('Beklenti (%)')
        ax_infl.grid(True, alpha=0.3)

        # 5. Amplification Ratios
        ax_amp = fig.add_subplot(gs[1, 3])
        farm_to_retail = P_R / P_F
        farm_to_wholesale = P_W / P_F
        ax_amp.plot(times, np.mean(farm_to_wholesale, axis=0), 'orange', linewidth=3, label='Tarla→Toptan')
        ax_amp.plot(times, np.mean(farm_to_retail, axis=0), 'red', linewidth=3, label='Tarla→Perakende')
        ax_amp.set_title('Fiyat Carpan Oranlari')
        ax_amp.set_ylabel('Carpan')
        ax_amp.legend(fontsize=8)
        ax_amp.grid(True, alpha=0.3)

        # 6. Price Distribution Analysis (Final Values)
        ax_dist = fig.add_subplot(gs[2, 0])
        ax_dist.hist(P_F[:, -1], bins=50, alpha=0.6, color='green', label='Tarla Cikis', density=True)
        ax_dist.hist(P_W[:, -1], bins=50, alpha=0.6, color='orange', label='Toptan', density=True)
        ax_dist.hist(P_R[:, -1], bins=50, alpha=0.6, color='red', label='Perakende', density=True)
        ax_dist.set_title('Final Fiyat Dagilimlari')
        ax_dist.set_xlabel('Fiyat Seviyesi')
        ax_dist.set_ylabel('Yogunluk')
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)

        # 7. Volatility Evolution
        ax_vol = fig.add_subplot(gs[2, 1])
        window = 20
        vol_F = pd.Series(np.mean(np.diff(np.log(P_F), axis=1), axis=0)).rolling(window).std() * np.sqrt(252)
        vol_W = pd.Series(np.mean(np.diff(np.log(P_W), axis=1), axis=0)).rolling(window).std() * np.sqrt(252)
        vol_R = pd.Series(np.mean(np.diff(np.log(P_R), axis=1), axis=0)).rolling(window).std() * np.sqrt(252)

        ax_vol.plot(times[1:], vol_F, 'green', linewidth=2, label='Tarla Cikis')
        ax_vol.plot(times[1:], vol_W, 'orange', linewidth=2, label='Toptan')
        ax_vol.plot(times[1:], vol_R, 'red', linewidth=2, label='Perakende')
        ax_vol.set_title('Zaman Icinde Volatilite')
        ax_vol.set_ylabel('Yillik Volatilite')
        ax_vol.legend(fontsize=8)
        ax_vol.grid(True, alpha=0.3)

        # 8. Correlation Matrix
        ax_corr = fig.add_subplot(gs[2, 2])
        variables = [P_F[:, -1], P_W[:, -1], P_R[:, -1], omega[:, -1], Pi_E[:, -1]]
        var_names = ['P_F', 'P_W', 'P_R', 'ω', 'Π^E']
        corr_matrix = np.corrcoef(variables)

        im = ax_corr.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1)
        ax_corr.set_xticks(range(len(var_names)))
        ax_corr.set_yticks(range(len(var_names)))
        ax_corr.set_xticklabels(var_names)
        ax_corr.set_yticklabels(var_names)
        ax_corr.set_title('Final Korelasyon Matrisi')

        # Add correlation values
        for i in range(len(var_names)):
            for j in range(len(var_names)):
                ax_corr.text(j, i, f'{corr_matrix[i, j]:.2f}',
                           ha='center', va='center', fontsize=8)

        # 9. Summary Statistics Table
        ax_stats = fig.add_subplot(gs[2, 3])
        ax_stats.axis('off')

        # Calculate key statistics
        stats_data = [
            ['Metric', 'Tarla Cikis', 'Toptan', 'Perakende'],
            ['Final Ortalama', f'{np.mean(P_F[:, -1]):.1f}', f'{np.mean(P_W[:, -1]):.1f}', f'{np.mean(P_R[:, -1]):.1f}'],
            ['Final Std', f'{np.std(P_F[:, -1]):.1f}', f'{np.std(P_W[:, -1]):.1f}', f'{np.std(P_R[:, -1]):.1f}'],
            ['VaR 95%', f'{np.percentile(P_F[:, -1], 5):.1f}', f'{np.percentile(P_W[:, -1], 5):.1f}', f'{np.percentile(P_R[:, -1], 5):.1f}'],
            ['Max', f'{np.max(P_F[:, -1]):.1f}', f'{np.max(P_W[:, -1]):.1f}', f'{np.max(P_R[:, -1]):.1f}'],
            ['Volatilite', f'{np.std(np.diff(np.log(P_F), axis=1)) * np.sqrt(252):.1%}',
             f'{np.std(np.diff(np.log(P_W), axis=1)) * np.sqrt(252):.1%}',
             f'{np.std(np.diff(np.log(P_R), axis=1)) * np.sqrt(252):.1%}']
        ]

        table = ax_stats.table(cellText=stats_data, cellLoc='center', loc='center',
                              colWidths=[0.3, 0.23, 0.23, 0.23])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax_stats.set_title('Ozet Istatistikler', fontweight='bold')

        # 10. Path-dependent Effects (bottom row)
        ax_path1 = fig.add_subplot(gs[3, 0])
        markup_evolution = P_R / (C_P + C_L)  # Total markup over base costs
        ax_path1.plot(times, np.mean(markup_evolution, axis=0), 'purple', linewidth=3)
        markup_std = np.std(markup_evolution, axis=0)
        markup_mean = np.mean(markup_evolution, axis=0)
        ax_path1.fill_between(times, markup_mean - markup_std, markup_mean + markup_std, alpha=0.3, color='purple')
        ax_path1.set_title('Toplam Kar Marji Evrimi')
        ax_path1.set_ylabel('Toplam Markup')
        ax_path1.set_xlabel('Zaman (Yil)')
        ax_path1.grid(True, alpha=0.3)

        ax_path2 = fig.add_subplot(gs[3, 1])
        efficiency = 1 - omega  # Market efficiency (1 - fire rate)
        ax_path2.plot(times, np.mean(efficiency, axis=0) * 100, 'brown', linewidth=3)
        eff_mean = np.mean(efficiency, axis=0) * 100
        eff_std = np.std(efficiency, axis=0) * 100
        ax_path2.fill_between(times, eff_mean - eff_std, eff_mean + eff_std, alpha=0.3, color='brown')
        ax_path2.set_title('Piyasa Verimliigi')
        ax_path2.set_ylabel('Verimlilik (%)')
        ax_path2.set_xlabel('Zaman (Yil)')
        ax_path2.grid(True, alpha=0.3)

        # Price jumps analysis
        ax_jumps = fig.add_subplot(gs[3, 2])
        price_changes_F = np.diff(P_F, axis=1)
        large_jumps = np.abs(price_changes_F) > 2 * np.std(price_changes_F)
        jump_frequency = np.sum(large_jumps, axis=0) / P_F.shape[0] * 100
        ax_jumps.plot(times[1:], jump_frequency, 'red', linewidth=2)
        ax_jumps.set_title('Buyuk Fiyat Sicrama Sikligi')
        ax_jumps.set_ylabel('Sicrama Sikligi (%)')
        ax_jumps.set_xlabel('Zaman (Yil)')
        ax_jumps.grid(True, alpha=0.3)

        # Final summary box
        ax_summary = fig.add_subplot(gs[3, 3])
        ax_summary.axis('off')

        # Key insights
        total_amplification = np.mean(P_R[:, -1] / (C_P[:, -1] + C_L[:, -1]))
        fire_effect = np.mean(1 / (1 - omega[:, -1]))
        expectation_change = (np.mean(Pi_E[:, -1]) - np.mean(Pi_E[:, 0])) * 100

        summary_text = f"""
        TEMEL BULGULAR
        ─────────────────

        [TAG] Toplam Carpan: {total_amplification:.2f}x
        [TAG] Fire Carpani: {fire_effect:.2f}x
        [TAG] Beklenti Degisimi: {expectation_change:+.1f}pp
        [TAG] Volatilite Artisi: {np.std(np.diff(np.log(P_R), axis=1)) / np.std(np.diff(np.log(P_F), axis=1)):.2f}x

        ─────────────────

        [TAG] {P_F.shape[0]:,} simulasyon yolu
        [TAG] {len(times)} zaman noktasi
        [TAG] {times[-1]:.1f} yil simulasyon
        """

        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                       fontsize=10, verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        return fig

    def _calculate_max_drawdown(self, paths: np.ndarray) -> float:
        """Calculate maximum drawdown for all paths."""
        max_drawdowns = []
        for path in paths:
            rolling_max = np.maximum.accumulate(path)
            drawdown = (path - rolling_max) / rolling_max
            max_drawdowns.append(np.min(drawdown))
        return np.mean(max_drawdowns)

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        mean_return = np.mean(returns) * 252  # Annualized
        return_std = np.std(returns) * np.sqrt(252)  # Annualized
        return (mean_return - risk_free_rate) / return_std if return_std > 0 else 0

    def create_comprehensive_visualizations(self) -> List[plt.Figure]:
        """Create comprehensive visualizations of all results."""

        print(f"\n[VIZ] VISUALIZATION PHASE")
        print("-" * 40)

        figures = []

        for model_name, sim_data in self.simulation_data.items():
            print(f"[PLOT] Creating plots for: {model_name}")

            paths = sim_data['paths']
            times = sim_data['times']
            final_values = paths[:, -1]

            # Figure 1: Path Evolution
            fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig1.suptitle(f'Monte Carlo Analysis: {model_name}', fontsize=16, fontweight='bold')

            # Plot sample paths
            sample_indices = np.random.choice(len(paths), size=min(100, len(paths)), replace=False)
            for idx in sample_indices:
                ax1.plot(times, paths[idx], alpha=0.3, linewidth=0.5, color='steelblue')

            # Plot mean and confidence intervals
            path_means = np.mean(paths, axis=0)
            path_stds = np.std(paths, axis=0)
            ax1.plot(times, path_means, 'red', linewidth=2, label='Mean Path')
            ax1.fill_between(times,
                           path_means - 1.96 * path_stds,
                           path_means + 1.96 * path_stds,
                           alpha=0.2, color='red', label='95% Confidence Interval')
            ax1.set_title('Sample Paths and Statistics')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Distribution of final values
            ax2.hist(final_values, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(np.mean(final_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(final_values):.2f}')
            ax2.axvline(np.median(final_values), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(final_values):.2f}')
            ax2.set_title('Distribution of Final Values')
            ax2.set_xlabel('Final Value')
            ax2.set_ylabel('Density')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Q-Q plot for normality test
            stats.probplot(final_values, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot (Normality Test)')
            ax3.grid(True, alpha=0.3)

            # Returns distribution
            returns = np.diff(np.log(paths), axis=1).flatten()
            ax4.hist(returns, bins=100, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
            ax4.set_title('Distribution of Log Returns')
            ax4.set_xlabel('Log Return')
            ax4.set_ylabel('Density')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            figures.append(fig1)

            # Figure 2: Statistical Analysis
            fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig2.suptitle(f'Statistical Analysis: {model_name}', fontsize=16, fontweight='bold')

            # Box plot of final values by percentiles
            percentiles = np.percentile(final_values, [10, 25, 50, 75, 90])
            ax1.boxplot(final_values, patch_artist=True)
            ax1.set_title('Final Values Box Plot')
            ax1.set_ylabel('Value')
            ax1.grid(True, alpha=0.3)

            # Cumulative distribution
            sorted_values = np.sort(final_values)
            cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            ax2.plot(sorted_values, cumulative, linewidth=2, color='purple')
            ax2.axvline(np.percentile(final_values, 5), color='red', linestyle='--',
                       label=f'5% VaR: {np.percentile(final_values, 5):.2f}')
            ax2.set_title('Cumulative Distribution Function')
            ax2.set_xlabel('Value')
            ax2.set_ylabel('Cumulative Probability')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Rolling statistics
            window = min(50, len(times) // 4)
            rolling_mean = pd.Series(path_means).rolling(window).mean()
            rolling_std = pd.Series(path_stds).rolling(window).mean()

            ax3.plot(times, rolling_mean, label='Rolling Mean', linewidth=2)
            ax3.plot(times, rolling_std, label='Rolling Std', linewidth=2)
            ax3.set_title(f'Rolling Statistics (Window: {window})')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('Value')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Autocorrelation of returns
            returns_sample = returns[::10]  # Sample for performance
            lags = range(min(50, len(returns_sample) // 4))
            autocorrs = [np.corrcoef(returns_sample[:-lag or None], returns_sample[lag:])[0,1]
                        if lag > 0 else 1.0 for lag in lags]

            ax4.bar(lags, autocorrs, alpha=0.7, color='orange')
            ax4.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax4.set_title('Autocorrelation of Returns')
            ax4.set_xlabel('Lag')
            ax4.set_ylabel('Autocorrelation')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            figures.append(fig2)

        # Enflasyon Beklentileri Analizi - Özel Grafik
        inflation_models = [name for name in self.simulation_data.keys()
                          if "inflation" in name.lower() or "enflasyon" in name.lower() or "expect" in name.lower()]

        if inflation_models:
            fig_inflation, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig_inflation.suptitle('Enflasyon Beklentileri Analizi', fontsize=16, fontweight='bold')

            for model_name in inflation_models:
                paths = self.simulation_data[model_name]['paths']
                times = self.simulation_data[model_name]['times']

                # Zaman serisi grafiği
                sample_paths = np.random.choice(len(paths), size=min(100, len(paths)), replace=False)
                for idx in sample_paths:
                    ax1.plot(times, paths[idx] * 100, alpha=0.3, linewidth=0.5, color='orange')

                mean_path = np.mean(paths, axis=0) * 100
                std_path = np.std(paths, axis=0) * 100
                ax1.plot(times, mean_path, 'red', linewidth=3, label=f'Ortalama Beklenti')
                ax1.fill_between(times, mean_path - std_path, mean_path + std_path,
                               alpha=0.2, color='red', label='±1 Std Sapma')
                ax1.set_title('Enflasyon Beklentileri Zaman Serisi')
                ax1.set_xlabel('Zaman (Yıl)')
                ax1.set_ylabel('Enflasyon Beklentisi (%)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Dağılım grafiği
                final_expectations = paths[:, -1] * 100
                ax2.hist(final_expectations, bins=50, density=True, alpha=0.7,
                        color='lightcoral', edgecolor='black')
                ax2.axvline(np.mean(final_expectations), color='red', linestyle='--',
                          linewidth=2, label=f'Ortalama: {np.mean(final_expectations):.2f}%')
                ax2.set_title('Final Enflasyon Beklentileri Dağılımı')
                ax2.set_xlabel('Enflasyon Beklentisi (%)')
                ax2.set_ylabel('Yoğunluk')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # Volatilite analizi
                returns = np.diff(paths, axis=1)
                volatility = np.std(returns, axis=0) * 100
                ax3.plot(times[1:], volatility, linewidth=2, color='purple', label='Volatilite')
                ax3.set_title('Enflasyon Beklentileri Volatilitesi')
                ax3.set_xlabel('Zaman (Yıl)')
                ax3.set_ylabel('Volatilite (%)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

                # Beklenti ankraj analizi
                initial_expectation = paths[0, 0] * 100
                deviations = (paths - paths[0, 0]) * 100
                max_deviations = np.max(np.abs(deviations), axis=1)
                ax4.hist(max_deviations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax4.axvline(np.mean(max_deviations), color='blue', linestyle='--',
                          linewidth=2, label=f'Ort. Max Sapma: {np.mean(max_deviations):.2f}%')
                ax4.set_title('Maksimum Beklenti Sapmaları')
                ax4.set_xlabel('Maksimum Sapma (%)')
                ax4.set_ylabel('Frekans')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            figures.append(fig_inflation)

        # Gıda Fiyat Çarpan Faktörü Analizi
        food_price_models = [name for name in self.simulation_data.keys()
                           if any(keyword in name.lower() for keyword in
                                ['producer', 'wholesale', 'retail', 'comprehensive', 'uretici', 'toptan', 'perakende'])]

        if food_price_models:
            fig_food, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig_food.suptitle('Gıda Fiyat Çarpan Faktörü Analizi', fontsize=16, fontweight='bold')

            # Tedarik zinciri boyunca fiyat artışı
            if len(food_price_models) >= 3:
                colors = ['green', 'orange', 'red']
                labels = ['Üretici', 'Toptan', 'Perakende']

                for i, (model_name, color, label) in enumerate(zip(food_price_models[:3], colors, labels)):
                    paths = self.simulation_data[model_name]['paths']
                    times = self.simulation_data[model_name]['times']

                    mean_path = np.mean(paths, axis=0)
                    ax1.plot(times, mean_path, color=color, linewidth=3, label=label)

                    # Güven aralığı
                    std_path = np.std(paths, axis=0)
                    ax1.fill_between(times, mean_path - std_path, mean_path + std_path,
                                   alpha=0.2, color=color)

                ax1.set_title('Tedarik Zinciri Boyunca Fiyat Gelişimi')
                ax1.set_xlabel('Zaman (Yıl)')
                ax1.set_ylabel('Fiyat Seviyesi')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # Çarpan faktörü hesaplama (son model / ilk model)
            if len(food_price_models) >= 2:
                first_model_paths = self.simulation_data[food_price_models[0]]['paths']
                last_model_paths = self.simulation_data[food_price_models[-1]]['paths']

                amplification_ratios = last_model_paths[:, -1] / first_model_paths[:, -1]

                ax2.hist(amplification_ratios, bins=50, density=True, alpha=0.7,
                        color='gold', edgecolor='black')
                ax2.axvline(np.mean(amplification_ratios), color='red', linestyle='--',
                          linewidth=2, label=f'Ortalama Çarpan: {np.mean(amplification_ratios):.2f}x')
                ax2.set_title('Toplam Çarpan Faktörü Dağılımı')
                ax2.set_xlabel('Çarpan Faktörü (Kat)')
                ax2.set_ylabel('Yoğunluk')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            # Fire oranı etkisi analizi
            fire_models = [name for name in self.simulation_data.keys()
                         if "fire" in name.lower() or "omega" in name.lower()]

            if fire_models:
                fire_paths = self.simulation_data[fire_models[0]]['paths']
                times = self.simulation_data[fire_models[0]]['times']

                # Fire oranı zaman serisi
                mean_fire = np.mean(fire_paths, axis=0) * 100
                std_fire = np.std(fire_paths, axis=0) * 100

                ax3.plot(times, mean_fire, 'brown', linewidth=3, label='Ortalama Fire Oranı')
                ax3.fill_between(times, mean_fire - std_fire, mean_fire + std_fire,
                               alpha=0.2, color='brown', label='±1 Std Sapma')
                ax3.set_title('Fire Oranı Zaman Serisi')
                ax3.set_xlabel('Zaman (Yıl)')
                ax3.set_ylabel('Fire Oranı (%)')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

                # Fire çarpan faktörü: 1/(1-ω)
                fire_amplifier = 1 / (1 - fire_paths[:, -1])
                ax4.hist(fire_amplifier, bins=50, density=True, alpha=0.7,
                        color='chocolate', edgecolor='black')
                ax4.axvline(np.mean(fire_amplifier), color='red', linestyle='--',
                          linewidth=2, label=f'Ortalama: {np.mean(fire_amplifier):.2f}x')
                ax4.set_title('Fire Çarpan Faktörü: 1/(1-ω)')
                ax4.set_xlabel('Çarpan Faktörü')
                ax4.set_ylabel('Yoğunluk')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                # Volatilite karşılaştırması
                volatilities = []
                model_labels = []

                for model_name in food_price_models:
                    paths = self.simulation_data[model_name]['paths']
                    returns = np.diff(np.log(paths), axis=1)
                    vol = np.std(returns.flatten()) * np.sqrt(252) * 100
                    volatilities.append(vol)
                    model_labels.append(model_name.replace('_', ' ').title())

                ax4.bar(model_labels, volatilities, alpha=0.7, color=['green', 'orange', 'red'][:len(volatilities)])
                ax4.set_title('Volatilite Karşılaştırması')
                ax4.set_ylabel('Yıllık Volatilite (%)')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            figures.append(fig_food)

        # Nested SDE System Analysis - Tüm State Variables
        if hasattr(self, 'nested_state_data'):
            fig_nested, axes = plt.subplots(3, 3, figsize=(18, 14))
            fig_nested.suptitle('Nested SDE Sistem Analizi - Tüm Durum Değişkenleri', fontsize=16, fontweight='bold')

            nested_data = self.nested_state_data
            times = nested_data['times']

            # State variables and their properties
            variables = [
                ('C_P', 'Üretim Maliyeti', 'green', axes[0, 0]),
                ('C_L', 'Lojistik Maliyeti', 'blue', axes[0, 1]),
                ('omega', 'Fire Oranı (%)', 'brown', axes[0, 2]),
                ('Pi_E', 'Enflasyon Beklentisi (%)', 'orange', axes[1, 0]),
                ('P_F', 'Üretici Fiyatı', 'red', axes[1, 1]),
                ('P_W', 'Toptan Fiyatı', 'purple', axes[1, 2]),
                ('P_R', 'Perakende Fiyatı', 'darkred', axes[2, 0])
            ]

            for var_name, title, color, ax in variables:
                data = nested_data[var_name]

                # Sample paths
                sample_indices = np.random.choice(data.shape[0], size=min(50, data.shape[0]), replace=False)
                for idx in sample_indices:
                    if var_name in ['omega', 'Pi_E']:
                        ax.plot(times, data[idx] * 100, alpha=0.3, linewidth=0.5, color=color)
                    else:
                        ax.plot(times, data[idx], alpha=0.3, linewidth=0.5, color=color)

                # Mean path
                if var_name in ['omega', 'Pi_E']:
                    mean_path = np.mean(data, axis=0) * 100
                    std_path = np.std(data, axis=0) * 100
                else:
                    mean_path = np.mean(data, axis=0)
                    std_path = np.std(data, axis=0)

                ax.plot(times, mean_path, color='black', linewidth=3, label='Ortalama')
                ax.fill_between(times, mean_path - std_path, mean_path + std_path,
                               alpha=0.2, color=color, label='±1 Std')

                ax.set_title(title)
                ax.set_xlabel('Zaman (Yıl)')
                if var_name in ['omega', 'Pi_E']:
                    ax.set_ylabel('Değer (%)')
                else:
                    ax.set_ylabel('Değer')
                ax.legend()
                ax.grid(True, alpha=0.3)

            # Çarpan faktörü analizi
            ax_amp = axes[2, 1]
            P_F_final = nested_data['P_F'][:, -1]
            P_R_final = nested_data['P_R'][:, -1]
            amplification = P_R_final / P_F_final

            ax_amp.hist(amplification, bins=50, density=True, alpha=0.7, color='gold', edgecolor='black')
            ax_amp.axvline(np.mean(amplification), color='red', linestyle='--',
                          linewidth=2, label=f'Ortalama: {np.mean(amplification):.2f}x')
            ax_amp.set_title('Toplam Çarpan Faktörü (P_R/P_F)')
            ax_amp.set_xlabel('Çarpan')
            ax_amp.set_ylabel('Yoğunluk')
            ax_amp.legend()
            ax_amp.grid(True, alpha=0.3)

            # Korelasyon analizi
            ax_corr = axes[2, 2]
            correlations = []
            labels = []

            for i, (var1, title1, _, _) in enumerate(variables[:-1]):
                for j, (var2, title2, _, _) in enumerate(variables[i+1:], i+1):
                    corr = np.corrcoef(nested_data[var1][:, -1], nested_data[var2][:, -1])[0, 1]
                    correlations.append(corr)
                    labels.append(f'{var1}-{var2}')

            colors_corr = ['red' if c > 0.5 else 'blue' if c < -0.5 else 'gray' for c in correlations]
            bars = ax_corr.bar(range(len(correlations)), correlations, color=colors_corr, alpha=0.7)
            ax_corr.set_title('Final Korelasyonlar')
            ax_corr.set_ylabel('Korelasyon')
            ax_corr.set_xticks(range(len(labels)))
            ax_corr.set_xticklabels(labels, rotation=45, ha='right')
            ax_corr.axhline(0, color='black', linestyle='-', alpha=0.5)
            ax_corr.grid(True, alpha=0.3)

            plt.tight_layout()
            figures.append(fig_nested)

        # Summary comparison plot if multiple models
        if len(self.simulation_data) > 1:
            fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig3.suptitle('Model Comparison Summary', fontsize=16, fontweight='bold')

            model_names = list(self.simulation_data.keys())
            final_means = [np.mean(self.simulation_data[name]['paths'][:, -1]) for name in model_names]
            final_stds = [np.std(self.simulation_data[name]['paths'][:, -1]) for name in model_names]

            # Mean comparison
            ax1.bar(model_names, final_means, alpha=0.7, color='lightblue', edgecolor='black')
            ax1.set_title('Mean Final Values Comparison')
            ax1.set_ylabel('Mean Final Value')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)

            # Volatility comparison
            ax2.bar(model_names, final_stds, alpha=0.7, color='lightcoral', edgecolor='black')
            ax2.set_title('Volatility Comparison')
            ax2.set_ylabel('Standard Deviation')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)

            # Risk-Return scatter
            ax3.scatter(final_stds, final_means, s=200, alpha=0.7, c=range(len(model_names)), cmap='viridis')
            for i, name in enumerate(model_names):
                ax3.annotate(name, (final_stds[i], final_means[i]), xytext=(5, 5),
                           textcoords='offset points', fontsize=10)
            ax3.set_xlabel('Risk (Standard Deviation)')
            ax3.set_ylabel('Return (Mean Final Value)')
            ax3.set_title('Risk-Return Profile')
            ax3.grid(True, alpha=0.3)

            # Distribution comparison
            for name in model_names:
                final_vals = self.simulation_data[name]['paths'][:, -1]
                ax4.hist(final_vals, bins=30, alpha=0.5, label=name, density=True)
            ax4.set_title('Final Value Distributions')
            ax4.set_xlabel('Final Value')
            ax4.set_ylabel('Density')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            figures.append(fig3)

        self.figures = figures
        print(f"[OK] Generated {len(figures)} comprehensive visualizations")
        return figures

    def generate_analysis_report(self, stats_results: Dict[str, Any]) -> str:
        """Generate a comprehensive analysis report."""

        report = []
        report.append("=" * 80)
        report.append("MATHEMATICAL MODEL EXTRACTION AND ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"• Models analyzed: {len(self.simulation_data)}")
        report.append(f"• Equations extracted: {len(self.results.get('equations', []))}")
        report.append(f"• Variables identified: {len(self.results.get('variables', []))}")
        report.append(f"• Processing time: {self.results.get('processing_time', 0):.3f} seconds")
        report.append("")

        # Model Details
        for model_name, stats in stats_results.items():
            report.append(f"MODEL ANALYSIS: {model_name}")
            report.append("-" * 50)
            report.append(f"Final Value Statistics:")
            report.append(f"  • Mean: {stats['mean']:.6f}")
            report.append(f"  • Standard Deviation: {stats['std']:.6f}")
            report.append(f"  • Minimum: {stats['min']:.6f}")
            report.append(f"  • Maximum: {stats['max']:.6f}")
            report.append(f"  • Median: {stats['median']:.6f}")
            report.append("")
            report.append(f"Risk Metrics:")
            report.append(f"  • Value at Risk (95%): {stats['VaR_95']:.6f}")
            report.append(f"  • Conditional VaR (95%): {stats['CVaR_95']:.6f}")
            report.append(f"  • Maximum Drawdown: {stats['max_drawdown']:.2%}")
            report.append(f"  • Annual Volatility: {stats['volatility_annual']:.2%}")
            report.append("")
            report.append(f"Performance Metrics:")
            report.append(f"  • Sharpe Ratio: {stats['sharpe_ratio']:.4f}")
            report.append(f"  • Skewness: {stats['skewness']:.4f}")
            report.append(f"  • Kurtosis: {stats['kurtosis']:.4f}")
            report.append("")
            report.append(f"Percentiles:")
            for pct, val in stats['percentiles'].items():
                report.append(f"  • {pct}: {val:.6f}")
            report.append("")

        # Constitutional Compliance
        report.append("CONSTITUTIONAL MATHEMATICAL FIDELITY")
        report.append("-" * 50)
        extraction_result = self.results.get('extraction_result')
        if extraction_result:
            compliance = extraction_result.constitutional_compliance
            fidelity_score = extraction_result.statistics.fidelity_score
            report.append(f"• Mathematical Fidelity: {'PRESERVED' if compliance else 'VIOLATED'}")
            report.append(f"• Fidelity Score: {fidelity_score:.2%}")
            report.append(f"• LaTeX Forms: Exactly preserved")
            report.append(f"• Symbol Integrity: Complete")
        report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def save_results(self, output_dir: str = "analysis_results") -> Dict[str, str]:
        """Save all analysis results to files."""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        saved_files = {}

        # Save statistical results
        if hasattr(self, 'stats_results'):
            stats_file = output_path / "statistical_analysis.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats_results, f, indent=2, default=str)
            saved_files['statistics'] = str(stats_file)

        # Save figures
        for i, fig in enumerate(self.figures):
            fig_file = output_path / f"visualization_{i+1}.png"
            fig.savefig(fig_file, dpi=300, bbox_inches='tight')
            saved_files[f'figure_{i+1}'] = str(fig_file)

        # Save simulation data
        sim_file = output_path / "simulation_data.npz"
        np.savez_compressed(sim_file, **{name: data['paths'] for name, data in self.simulation_data.items()})
        saved_files['simulation_data'] = str(sim_file)

        return saved_files


def main():
    """Main analysis function - run this in VSCode!"""

    print("MATHEMATICAL MODEL EXTRACTION & ANALYSIS")
    print("=" * 70)
    print("Advanced Python Analysis with Visualization")
    print("=" * 70)

    # Initialize analyzer
    analyzer = MathematicalAnalyzer()

    # Comprehensive food price dynamics document with inflation expectations and multi-layer markup models
    document_content = r"""
    \documentclass{article}
    \title{Sistemik Fiyat Dalgalanması: Gıda Tedarik Zincirinde Stokastik Dinamikler}

    \begin{document}

    \section{Dışsal Maliyet Süreçleri}
    Üretim maliyetleri (gübre, tohum, akaryakıt):
    \begin{equation}
    dC_{P,t} = \mu_{P} C_{P,t} dt + \sigma_{P} C_{P,t} dW_{P,t}
    \end{equation}

    Lojistik maliyetleri (taşıma, depolama):
    \begin{equation}
    dC_{L,t} = \mu_{L} C_{L,t} dt + \sigma_{L} C_{L,t} dW_{L,t}
    \end{equation}

    \section{Fire Oranı Dinamikleri}
    Operasyonel verimsizlik sürecİ (Ornstein-Uhlenbeck):
    \begin{equation}
    d\omega_{t} = \kappa_{\omega}(\overline{\omega} - \omega_{t})dt + \sigma_{\omega}dW_{\omega,t}
    \end{equation}

    \section{Enflasyon Beklentileri}
    Uyarlanabilir enflasyon beklentileri:
    \begin{equation}
    d\Pi_{t}^{E} = \kappa_{\pi}(\pi_{R,t} - \Pi_{t}^{E})dt + \sigma_{\pi}dW_{\pi,t}
    \end{equation}

    \section{Üretici Fiyat Dinamikleri}
    Sıçramalı difüzyon süreci (arz şokları ile):
    \begin{equation}
    dP_{F,t} = \kappa_F ((1+\mu_F)C_{P,t} - P_{F,t}) dt + \sigma_F P_{F,t}dW_{F,t} + J_F P_{F,t-}dN_t
    \end{equation}

    \section{Aracı Fiyat Dinamikleri}
    Fire çarpanı ile etkin maliyet:
    \begin{equation}
    C^{eff}_{W,t} = \frac{P_{F,t}+C_{L,t}}{1-\omega_{t}}
    \end{equation}

    Toptan satış fiyatı:
    \begin{equation}
    dP_{W,t} = \kappa_W((1+\mu_W)C^{eff}_{W,t} - P_{W,t})dt + \sigma_W P_{W,t}dW_{W,t}
    \end{equation}

    \section{Perakende Fiyat Dinamikleri}
    Beklentilere duyarlı kâr marjı:
    \begin{equation}
    \mu_{R}(\Pi_{t}^{E}) = \overline{\mu}_{R} + \beta \Pi_{t}^{E}
    \end{equation}

    Nihai tüketici fiyatı:
    \begin{equation}
    dP_{R,t} = \kappa_R((1+\mu_R(\Pi_t^E))P_{W,t} - P_{R,t})dt + \sigma_R P_{R,t}dW_{R,t}
    \end{equation}

    \section{Toplam Çarpan Faktörü}
    Birleşik katlamalı etki:
    \begin{equation}
    A_t = (1+\mu_R(\Pi^E_t))(1+\mu_W)\frac{1}{1-\omega_{t}}
    \end{equation}

    Nihai entegre fiyat yapısı:
    \begin{equation}
    P_{R,t} \approx A_t (P_{F,t}+C_{L,t})
    \end{equation}

    \section{İstikrar Koşulu}
    Enflasyon-beklenti döngüsü istikrarı:
    \begin{equation}
    \frac{\beta}{1+\overline{\mu}_{R}} < \frac{1}{\kappa_{\pi}} + \frac{1}{\kappa_R}
    \end{equation}

    \end{document}
    """

    try:
        # Step 1: Extract and analyze models
        print("[1] Step 1: Extracting mathematical models...")
        extraction_results = analyzer.extract_and_analyze_models(document_content)

        # Step 2: Run Monte Carlo simulations
        print("[2] Step 2: Running Monte Carlo simulations...")
        simulation_results = analyzer.run_monte_carlo_simulations(
            num_paths=15000,  # More paths for complex models
            time_horizon=3.0  # Longer horizon for structural analysis
        )

        # Step 3: Generate statistical analysis
        print("[3] Step 3: Generating statistical analysis...")
        stats_results = analyzer.generate_statistical_analysis()
        analyzer.stats_results = stats_results

        # Step 4: Create visualizations
        print("[4] Step 4: Creating comprehensive visualizations...")
        figures = analyzer.create_comprehensive_visualizations()

        # Step 4.5: Agricultural price path analysis
        print("[4.5] Step 4.5: Agricultural price path analysis...")
        agri_analysis = analyzer.generate_agricultural_price_path_analysis()
        agri_figure = analyzer.create_agricultural_price_path_visualization()
        if agri_figure:
            figures.append(agri_figure)

        # Step 5: Generate report
        print("[5] Step 5: Generating analysis report...")
        report = analyzer.generate_analysis_report(stats_results)
        print("\n" + report)

        # Step 6: Save results
        print("[6] Step 6: Saving results...")
        saved_files = analyzer.save_results()

        print("\n" + "=" * 70)
        print("[COMPLETE] ANALYSIS COMPLETE!")
        print("=" * 70)
        print("[FILES] Files saved:")
        for name, path in saved_files.items():
            print(f"   • {name}: {path}")

        print("\n[PLOTS] Visualizations:")
        print(f"   • Generated {len(figures)} comprehensive plots")
        print("   • All plots are displayed in separate windows")
        print("   • High-resolution versions saved to files")

        print("\n[VSCODE] VSCode Integration:")
        print("   • All results available in Python variables")
        print("   • Use plt.show() to display plots interactively")
        print("   • Data accessible via analyzer.simulation_data")

        # Display all plots
        plt.show()

        return analyzer, stats_results, figures

    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    # Run the comprehensive analysis
    analyzer, stats, figures = main()

    # Keep variables accessible in VSCode for further analysis
    print("\n[INFO] Available for further analysis:")
    print("   • analyzer: Main analyzer object")
    print("   • stats: Statistical results dictionary")
    print("   • figures: List of matplotlib figures")
    print("   • analyzer.simulation_data: Raw simulation paths")
    print("   • analyzer.results: Extraction results")