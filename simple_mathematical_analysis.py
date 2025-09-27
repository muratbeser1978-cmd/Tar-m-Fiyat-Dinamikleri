#!/usr/bin/env python3
"""
Simple Mathematical Analysis - Direct SDE Simulation
===================================================

Direct implementation of SDE simulations without complex validation.
Perfect for VSCode with immediate results and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats as scipy_stats
from datetime import datetime
import warnings

# Configure plotting
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class SimpleSDE:
    """Simple SDE simulation without complex validation."""

    def __init__(self, name, params, color='blue'):
        self.name = name
        self.params = params
        self.color = color
        self.paths = None
        self.times = None

    def simulate(self, T=1.0, N=252, M=5000, seed=42):
        """Simulate SDE paths."""
        np.random.seed(seed)
        dt = T / N
        self.times = np.linspace(0, T, N + 1)

        # Random increments
        dW = np.random.normal(0, np.sqrt(dt), (M, N))

        # Initialize paths
        self.paths = np.zeros((M, N + 1))

        if 'gbm' in self.name.lower():
            self._simulate_gbm(dW, dt)
        elif 'ou' in self.name.lower():
            self._simulate_ou(dW, dt)
        elif 'cir' in self.name.lower():
            self._simulate_cir(dW, dt)

        return self.paths, self.times

    def _simulate_gbm(self, dW, dt):
        """Geometric Brownian Motion: dS = μS dt + σS dW"""
        mu = self.params.get('mu', 0.05)
        sigma = self.params.get('sigma', 0.2)
        S0 = self.params.get('S0', 100.0)

        self.paths[:, 0] = S0
        for i in range(len(dW[0])):
            self.paths[:, i+1] = self.paths[:, i] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * dW[:, i]
            )

    def _simulate_ou(self, dW, dt):
        """Ornstein-Uhlenbeck: dX = κ(θ - X)dt + σ dW"""
        kappa = self.params.get('kappa', 2.0)
        theta = self.params.get('theta', 0.05)
        sigma = self.params.get('sigma', 0.1)
        X0 = self.params.get('X0', 0.03)

        self.paths[:, 0] = X0
        for i in range(len(dW[0])):
            drift = kappa * (theta - self.paths[:, i]) * dt
            diffusion = sigma * dW[:, i]
            self.paths[:, i+1] = self.paths[:, i] + drift + diffusion

    def _simulate_cir(self, dW, dt):
        """Cox-Ingersoll-Ross: dr = κ(θ - r)dt + σ√r dW"""
        kappa = self.params.get('kappa', 2.0)
        theta = self.params.get('theta', 0.05)
        sigma = self.params.get('sigma', 0.1)
        r0 = self.params.get('r0', 0.03)

        self.paths[:, 0] = r0
        for i in range(len(dW[0])):
            drift = kappa * (theta - self.paths[:, i]) * dt
            diffusion = sigma * np.sqrt(np.maximum(self.paths[:, i], 0)) * dW[:, i]
            self.paths[:, i+1] = np.maximum(self.paths[:, i] + drift + diffusion, 0)

    def get_statistics(self):
        """Calculate comprehensive statistics."""
        if self.paths is None:
            return {}

        final_values = self.paths[:, -1]
        returns = np.diff(np.log(self.paths), axis=1).flatten()

        # Basic statistics
        stats_dict = {
            'mean': np.mean(final_values),
            'std': np.std(final_values),
            'min': np.min(final_values),
            'max': np.max(final_values),
            'median': np.median(final_values),
            'skewness': scipy_stats.skew(final_values),
            'kurtosis': scipy_stats.kurtosis(final_values),
            'var': np.var(final_values)
        }

        # Percentiles
        percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])
        stats_dict['percentiles'] = {
            '5%': percentiles[0],
            '25%': percentiles[1],
            '50%': percentiles[2],
            '75%': percentiles[3],
            '95%': percentiles[4]
        }

        # Risk metrics
        VaR_95 = np.percentile(final_values, 5)
        CVaR_95 = np.mean(final_values[final_values <= VaR_95])

        # Performance metrics
        annual_return = np.mean(returns) * 252
        annual_vol = np.std(returns) * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0

        # Maximum drawdown
        rolling_max = np.maximum.accumulate(self.paths, axis=1)
        drawdowns = (self.paths - rolling_max) / rolling_max
        max_dd = np.mean(np.min(drawdowns, axis=1))

        stats_dict.update({
            'VaR_95': VaR_95,
            'CVaR_95': CVaR_95,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'final_values': final_values,
            'returns': returns
        })

        return stats_dict

def create_sde_models():
    """Create SDE models for analysis."""
    models = []

    # Geometric Brownian Motion
    models.append(SimpleSDE(
        name="Geometric Brownian Motion (GBM)",
        params={'mu': 0.08, 'sigma': 0.25, 'S0': 100.0},
        color='steelblue'
    ))

    # Ornstein-Uhlenbeck Process
    models.append(SimpleSDE(
        name="Ornstein-Uhlenbeck (OU)",
        params={'kappa': 3.0, 'theta': 0.05, 'sigma': 0.12, 'X0': 0.03},
        color='crimson'
    ))

    # Cox-Ingersoll-Ross Model
    models.append(SimpleSDE(
        name="Cox-Ingersoll-Ross (CIR)",
        params={'kappa': 2.5, 'theta': 0.04, 'sigma': 0.15, 'r0': 0.02},
        color='forestgreen'
    ))

    return models

def run_comprehensive_analysis():
    """Run comprehensive SDE analysis with visualizations."""

    print("MATHEMATICAL SDE ANALYSIS")
    print("=" * 60)
    print("Direct SDE Simulation with Statistical Analysis")
    print("=" * 60)

    # Create and simulate models
    models = create_sde_models()

    print("\nSIMULATION PHASE")
    print("-" * 40)

    simulation_results = {}

    for model in models:
        print(f"Simulating: {model.name}")

        # Run simulation
        paths, times = model.simulate(T=2.0, N=504, M=10000, seed=42)
        stats = model.get_statistics()

        simulation_results[model.name] = {
            'model': model,
            'paths': paths,
            'times': times,
            'stats': stats
        }

        print(f"   Completed: {len(paths)} paths over {len(times)} time steps")
        print(f"   Mean final value: {stats['mean']:.4f}")
        print(f"   Standard deviation: {stats['std']:.4f}")
        print(f"   Sharpe ratio: {stats['sharpe_ratio']:.4f}")

    # Create comprehensive visualizations
    print("\nVISUALIZATION PHASE")
    print("-" * 40)

    # Figure 1: Path Evolution and Distributions
    fig1, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig1.suptitle('SDE Path Analysis and Final Value Distributions', fontsize=16, fontweight='bold')

    for i, (name, result) in enumerate(simulation_results.items()):
        model = result['model']
        paths = result['paths']
        times = result['times']
        stats = result['stats']

        # Top row: Sample paths
        ax_paths = axes[0, i]

        # Plot sample paths
        n_sample = min(100, len(paths))
        sample_indices = np.random.choice(len(paths), n_sample, replace=False)
        for idx in sample_indices:
            ax_paths.plot(times, paths[idx], alpha=0.3, linewidth=0.5, color=model.color)

        # Plot mean path and confidence intervals
        path_mean = np.mean(paths, axis=0)
        path_std = np.std(paths, axis=0)
        ax_paths.plot(times, path_mean, color='black', linewidth=2.5, label='Mean')
        ax_paths.fill_between(times,
                             path_mean - 1.96 * path_std,
                             path_mean + 1.96 * path_std,
                             alpha=0.2, color=model.color, label='95% CI')

        ax_paths.set_title(f'{model.name}\nSample Paths', fontweight='bold')
        ax_paths.set_xlabel('Time (years)')
        ax_paths.set_ylabel('Value')
        ax_paths.legend()
        ax_paths.grid(True, alpha=0.3)

        # Bottom row: Final value distributions
        ax_dist = axes[1, i]
        final_vals = stats['final_values']

        ax_dist.hist(final_vals, bins=60, density=True, alpha=0.7,
                    color=model.color, edgecolor='black', linewidth=0.5)

        # Add statistical lines
        ax_dist.axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                       label=f"Mean: {stats['mean']:.3f}")
        ax_dist.axvline(stats['median'], color='orange', linestyle='--', linewidth=2,
                       label=f"Median: {stats['median']:.3f}")
        ax_dist.axvline(stats['VaR_95'], color='purple', linestyle='--', linewidth=2,
                       label=f"VaR(5%): {stats['VaR_95']:.3f}")

        ax_dist.set_title(f'{model.name}\nFinal Value Distribution', fontweight='bold')
        ax_dist.set_xlabel('Final Value')
        ax_dist.set_ylabel('Density')
        ax_dist.legend(fontsize=9)
        ax_dist.grid(True, alpha=0.3)

    plt.tight_layout()

    # Figure 2: Statistical Analysis
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Comprehensive Statistical Analysis', fontsize=16, fontweight='bold')

    # Collect data for comparison
    model_names = list(simulation_results.keys())
    model_short = [name.split('(')[1].replace(')', '') for name in model_names]
    colors = [simulation_results[name]['model'].color for name in model_names]

    means = [simulation_results[name]['stats']['mean'] for name in model_names]
    stds = [simulation_results[name]['stats']['std'] for name in model_names]
    sharpes = [simulation_results[name]['stats']['sharpe_ratio'] for name in model_names]
    max_dds = [abs(simulation_results[name]['stats']['max_drawdown']) for name in model_names]

    # Mean comparison
    bars1 = axes[0,0].bar(model_short, means, color=colors, alpha=0.8, edgecolor='black')
    axes[0,0].set_title('Mean Final Values Comparison', fontweight='bold')
    axes[0,0].set_ylabel('Mean Final Value')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars1, means):
        axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Volatility comparison
    bars2 = axes[0,1].bar(model_short, stds, color=colors, alpha=0.8, edgecolor='black')
    axes[0,1].set_title('Volatility (Standard Deviation)', fontweight='bold')
    axes[0,1].set_ylabel('Standard Deviation')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)

    for bar, val in zip(bars2, stds):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                      f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

    # Risk-Return scatter plot
    scatter = axes[1,0].scatter(stds, means, s=300, c=colors, alpha=0.8,
                               edgecolors='black', linewidth=2)

    for i, (short_name, x, y) in enumerate(zip(model_short, stds, means)):
        axes[1,0].annotate(short_name, (x, y), xytext=(8, 8),
                          textcoords='offset points', fontweight='bold',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    axes[1,0].set_xlabel('Risk (Standard Deviation)')
    axes[1,0].set_ylabel('Return (Mean Final Value)')
    axes[1,0].set_title('Risk-Return Profile', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)

    # Performance metrics
    x_pos = np.arange(len(model_short))
    width = 0.35

    bars3 = axes[1,1].bar(x_pos - width/2, sharpes, width, label='Sharpe Ratio',
                         color='lightblue', alpha=0.8, edgecolor='black')
    bars4 = axes[1,1].bar(x_pos + width/2, [dd*100 for dd in max_dds], width,
                         label='Max DD (%)', color='lightcoral', alpha=0.8, edgecolor='black')

    axes[1,1].set_title('Performance Metrics', fontweight='bold')
    axes[1,1].set_ylabel('Value')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(model_short, rotation=45)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Figure 3: Detailed Statistical Comparison
    fig3, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig3.suptitle('Detailed Statistical Analysis', fontsize=16, fontweight='bold')

    # Box plots of final values
    final_values_list = [simulation_results[name]['stats']['final_values'] for name in model_names]
    box_plot = axes[0,0].boxplot(final_values_list, labels=model_short, patch_artist=True)

    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    axes[0,0].set_title('Final Values Box Plot Comparison', fontweight='bold')
    axes[0,0].set_ylabel('Final Value')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)

    # Returns distribution overlay
    for name, color, short_name in zip(model_names, colors, model_short):
        returns = simulation_results[name]['stats']['returns']
        axes[0,1].hist(returns, bins=50, alpha=0.5, density=True,
                      color=color, label=short_name, edgecolor='black', linewidth=0.5)

    axes[0,1].set_title('Returns Distribution Overlay', fontweight='bold')
    axes[0,1].set_xlabel('Log Returns')
    axes[0,1].set_ylabel('Density')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Cumulative distributions
    for name, color, short_name in zip(model_names, colors, model_short):
        final_vals = simulation_results[name]['stats']['final_values']
        sorted_vals = np.sort(final_vals)
        cumulative = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        axes[1,0].plot(sorted_vals, cumulative, linewidth=2.5, color=color, label=short_name)

    axes[1,0].set_title('Cumulative Distribution Functions', fontweight='bold')
    axes[1,0].set_xlabel('Value')
    axes[1,0].set_ylabel('Cumulative Probability')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Q-Q plots for normality
    colors_cycle = ['blue', 'red', 'green']
    for i, (name, color_name) in enumerate(zip(model_names, colors_cycle)):
        final_vals = simulation_results[name]['stats']['final_values']
        scipy_stats.probplot(final_vals, dist="norm", plot=axes[1,1])
        line = axes[1,1].get_lines()[-1]
        line.set_color(colors[i])
        line.set_linewidth(2)

    axes[1,1].set_title('Q-Q Plots (Normality Test)', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Print detailed statistics
    print("\nDETAILED STATISTICAL RESULTS")
    print("=" * 80)

    for name in model_names:
        stats = simulation_results[name]['stats']
        print(f"\n{name}")
        print("-" * 50)
        print(f"Final Value Statistics:")
        print(f"  • Mean: {stats['mean']:.6f}")
        print(f"  • Standard Deviation: {stats['std']:.6f}")
        print(f"  • Minimum: {stats['min']:.6f}")
        print(f"  • Maximum: {stats['max']:.6f}")
        print(f"  • Median: {stats['median']:.6f}")
        print(f"  • Skewness: {stats['skewness']:.4f}")
        print(f"  • Kurtosis: {stats['kurtosis']:.4f}")

        print(f"\nRisk Metrics:")
        print(f"  • VaR (95%): {stats['VaR_95']:.6f}")
        print(f"  • CVaR (95%): {stats['CVaR_95']:.6f}")
        print(f"  • Maximum Drawdown: {abs(stats['max_drawdown']):.2%}")

        print(f"\nPerformance Metrics:")
        print(f"  • Annual Return: {stats['annual_return']:.2%}")
        print(f"  • Annual Volatility: {stats['annual_volatility']:.2%}")
        print(f"  • Sharpe Ratio: {stats['sharpe_ratio']:.4f}")

        print(f"\nPercentiles:")
        for pct, val in stats['percentiles'].items():
            print(f"  • {pct}: {val:.6f}")

    # Show all plots
    plt.show()

    print(f"\nANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Generated 3 comprehensive visualization figures")
    print(f"Analyzed {len(models)} different SDE models")
    print(f"Simulated {10000} Monte Carlo paths each")
    print(f"Analysis completed at: {datetime.now().strftime('%H:%M:%S')}")

    return simulation_results

if __name__ == "__main__":
    # Run the comprehensive analysis
    results = run_comprehensive_analysis()

    print(f"\nAVAILABLE FOR FURTHER ANALYSIS:")
    print("-" * 40)
    print(f"• results: Complete simulation results dictionary")
    print(f"• results[model_name]['paths']: Simulation paths")
    print(f"• results[model_name]['stats']: Statistical analysis")
    print(f"• results[model_name]['model']: SDE model object")
    print(f"\nExample usage:")
    print(f"  gbm_paths = results['Geometric Brownian Motion (GBM)']['paths']")
    print(f"  ou_stats = results['Ornstein-Uhlenbeck (OU)']['stats']")
    print(f"  plt.figure(); plt.plot(gbm_paths[0]); plt.show()  # Plot first path")