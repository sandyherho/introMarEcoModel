#!/usr/bin/python3

"""
biomass_models2.py: Implementation of Nutrient-Controlled Growth Rate Model

This script implements and visualizes the nutrient-controlled growth rate model 
described in Chapter 2.1.1 of 'Introduction to the Modelling of Marine Ecosystems' 
(Fennel & Neumann, 2014), specifically reproducing Figure 2.3.

Author: Sandy H. S. Herho <sandy.herho@email.ucr.edu>
Date: November 8, 2024
Version: 1.0.0
License: MIT

Book Examples
------------
To reproduce Figure 2.3 from the book:
    python biomass_models2.py --t_max=8.0 --N0=5.0 --P0=0.5 --k=0.2
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import sys
from datetime import datetime
import warnings
import argparse

class ParameterError(Exception):
    """Exception raised for invalid model parameters."""
    pass

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(description='Nutrient-Controlled Growth Rate Model')
    parser.add_argument('--t_max', type=float, default=8.0,
                      help='Maximum simulation time in days')
    parser.add_argument('--N0', type=float, default=5.0,
                      help='Initial nutrient concentration (mmol/m³)')
    parser.add_argument('--P0', type=float, default=0.5,
                      help='Initial phytoplankton concentration (mmol/m³)')
    parser.add_argument('--k', type=float, default=0.2,
                      help='Growth rate parameter (1/day/mmol N/m³)')
    return parser.parse_args()

def validate_parameters(params: Dict[str, float]) -> None:
    """
    Validates the input parameters for the model.
    
    Parameters
    ----------
    params : dict
        Dictionary containing model parameters
        
    Raises
    ------
    ParameterError
        If any parameter is invalid
    """
    # Only validate numerical parameters
    numerical_params = ['t_max', 'dt', 'N0', 'P0', 'k']
    for param_name in numerical_params:
        if param_name in params:
            value = params[param_name]
            if not isinstance(value, (int, float)) or value <= 0:
                raise ParameterError(
                    f"{param_name} must be a positive number. Got: {value}"
                )

def simulate_nutrient_model(
    t_max: float = 8.0,
    dt: float = 0.01,
    N0: float = 5.0,
    P0: float = 0.5,
    k: float = 0.2,
    save_path: Optional[str] = None,
    fig_size: Tuple[float, float] = (8, 6),
    dpi: int = 300,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Simulates and visualizes the nutrient-controlled growth rate model.
    """
    try:
        # Validate parameters
        params_to_validate = {'t_max': t_max, 'dt': dt, 'N0': N0, 'P0': P0, 'k': k}
        validate_parameters(params_to_validate)
        
        # Set custom plotting style
        plt.style.use('bmh')
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'serif',
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.spines.right': True,
            'axes.spines.top': True,
            'axes.grid': True,
            'grid.linestyle': ':',
            'grid.alpha': 0.3
        })
        
        # Create time array
        t = np.arange(0, t_max, dt)
        
        # Initialize arrays
        N = np.zeros_like(t)
        P = np.zeros_like(t)
        
        # Set initial conditions
        N[0] = N0
        P[0] = P0
        
        # Time integration
        for i in range(1, len(t)):
            # Calculate rates
            growth_rate = k * N[i-1] * P[i-1]
            
            # Update concentrations
            P[i] = P[i-1] + growth_rate * dt
            N[i] = N[i-1] - growth_rate * dt
            
            # Check for numerical instabilities
            if np.isnan(P[i]) or np.isnan(N[i]):
                raise RuntimeWarning("Numerical instability detected")
        
        # Create figure
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
        
        # Plot results
        ax.plot(t, N, 'b-', label='N', linewidth=1.5)
        ax.plot(t, P, 'r--', label='P', linewidth=1.5)
        
        # Add reference lines
        ax.axhline(y=P0, color='k', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.axhline(y=N0, color='k', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.axhline(y=P0+N0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Add labels
        ax.text(-0.3, P0, 'P₀', verticalalignment='center')
        ax.text(-0.3, N0, 'N₀', verticalalignment='center')
        ax.text(-0.3, P0+N0, 'P₀ + N₀', verticalalignment='center')
        
        # Configure axes
        ax.set_xlabel('t (d)', fontsize=12)
        ax.set_ylabel('N and P', fontsize=12)
        ax.legend(frameon=True, loc='best')
        
        # Set axis limits
        ax.set_xlim(-0.5, t_max)
        ax.set_ylim(-0.2, N0+P0+0.5)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(f"{save_path}.png", dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Figure saved as {save_path}.png")
        
        return fig, ax
        
    except ParameterError as e:
        print(f"Error in model parameters: {str(e)}")
        sys.exit(1)
    except RuntimeWarning as w:
        print(f"Warning in simulation: {str(w)}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)

def main():
    """Main execution function."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Run simulation with parsed arguments
        fig, ax = simulate_nutrient_model(
            t_max=args.t_max,
            N0=args.N0,
            P0=args.P0,
            k=args.k,
            save_path="fig2_3"
        )
        
        plt.show()
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
