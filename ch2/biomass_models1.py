#!/usr/bin/python3

"""
biomass_models1.py: Implementation of Biomass Models from Marine Ecosystem Modeling

This script implements and visualizes the biomass models described in Chapter 2.1.1 of
'Introduction to the Modelling of Marine Ecosystems' (Fennel & Neumann, 2014).

Author: Sandy H. S. Herho <sandy.herho@email.ucr.edu>
Date: November 8, 2024
Version: 1.0.0
License: MIT

Book Examples
------------
To reproduce Figure 2.2 from the book:
    python biomass_models1.py --t_max=6.0 --S0=5.0 --k=1.0
    
Parameters
----------
t_max : float
    Maximum simulation time in days
S0 : float
    Initial substrate/nutrient concentration (mmol/m³)
k : float
    Rate constant (1/day)

Output
------
- Displays the figure on screen
- Saves as 'fig2_2.png' in the current directory
- Both subplots show the dynamics of substrate/product and nutrient/phytoplankton
"""

# Required libraries
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import sys
from datetime import datetime
import warnings
import argparse

# Custom exception for parameter validation
class ParameterError(Exception):
    """Exception raised for invalid model parameters."""
    pass

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with default values."""
    parser = argparse.ArgumentParser(description='Biomass Models Simulation')
    parser.add_argument('--t_max', type=float, default=6.0,
                      help='Maximum simulation time in days')
    parser.add_argument('--S0', type=float, default=5.0,
                      help='Initial substrate concentration (mmol/m³)')
    parser.add_argument('--k', type=float, default=1.0,
                      help='Rate constant (1/day)')
    return parser.parse_args()

def validate_parameters(params: Dict[str, float]) -> None:
    """
    Validates the input parameters for the biomass models.
    
    Parameters
    ----------
    params : dict
        Dictionary containing model parameters
        
    Raises
    ------
    ParameterError
        If any parameter is invalid
    """
    # Check for negative values
    for param_name in ['t_max', 'dt', 'S0', 'P0', 'N0']:
        if params[param_name] <= 0:
            raise ParameterError(
                f"{param_name} must be positive. Got: {params[param_name]}"
            )
    
    # Check time step size
    if params['dt'] >= params['t_max']:
        raise ParameterError(
            f"Time step (dt={params['dt']}) must be smaller than t_max={params['t_max']}"
        )
    
    # Check rate constant
    if params['k'] <= 0:
        raise ParameterError(
            f"Rate constant k must be positive. Got: {params['k']}"
        )

def set_plotting_style():
    """Configure matplotlib to match the book's style."""
    plt.style.use('bmh')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'axes.spines.right': True,
        'axes.spines.top': True,
        'axes.grid': True,
        'grid.linestyle': ':',
        'grid.alpha': 0.5
    })

def simulate_biomass_models(
    t_max: float = 6.0,      # Maximum simulation time [days]
    dt: float = 0.01,        # Time step size [days]
    S0: float = 5.0,         # Initial substrate concentration [mmol/m³]
    P0: float = 0.5,         # Initial product/phytoplankton concentration [mmol/m³]
    N0: float = 5.0,         # Initial nutrient concentration [mmol/m³]
    k: float = 1.0,          # Rate constant [1/day]
    save_path: Optional[str] = None,  # Path to save figure
    fig_size: Tuple[float, float] = (8, 10),  # Figure size in inches
    dpi: int = 300,          # DPI for saved figure
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Simulates and visualizes two biomass models from marine ecosystem modeling.
    
    Parameters
    ----------
    t_max : float
        Maximum simulation time in days
    dt : float
        Time step size in days
    S0, P0, N0 : float
        Initial concentrations for substrate, product, and nutrient
    k : float
        Rate constant
    save_path : str, optional
        Path to save the figure
    fig_size : tuple
        Figure dimensions
    dpi : int
        Resolution for saved figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    ax : numpy.ndarray
        Array of the two subplot axes
    """
    try:
        # Validate parameters
        params = locals()
        validate_parameters({k: v for k, v in params.items() 
                           if k in ['t_max', 'dt', 'S0', 'P0', 'N0', 'k']})
        
        # Set custom plotting style
        set_plotting_style()
        
        # Create time array
        t = np.arange(0, t_max, dt)
        
        # First-order chemical reaction
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            S = S0 * np.exp(-k * t)
            P_chem = S0 * (1 - np.exp(-k * t))
        
        # Simple nutrient uptake model
        N = np.zeros_like(t)
        P_bio = np.zeros_like(t)
        
        # Initial conditions
        N[0] = N0
        P_bio[0] = P0
        
        # Time integration using Euler method
        for i in range(1, len(t)):
            dP = k * P_bio[i-1]
            dN = -k * P_bio[i-1]
            
            P_bio[i] = P_bio[i-1] + dP * dt
            N[i] = N[i-1] + dN * dt
            
            # Check for numerical instabilities
            if np.isnan(P_bio[i]) or np.isnan(N[i]):
                raise RuntimeWarning("Numerical instability detected")
        
        # Create figure
        fig, ax = plt.subplots(2, 1, figsize=fig_size, height_ratios=[1, 1])
        
        # First subplot: Chemical reaction
        ax[0].plot(t, S, 'b-', label='S', linewidth=2)
        ax[0].plot(t, P_chem, 'r--', label='P', linewidth=2)
        ax[0].set_xlabel('t/d', fontsize=14)
        ax[0].set_ylabel('S and P', fontsize=14)
        ax[0].legend(frameon=True)
        ax[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Second subplot: Nutrient uptake
        ax[1].plot(t, N, 'b-', label='N', linewidth=2)
        ax[1].plot(t, P_bio, 'r--', label='P', linewidth=2)
        ax[1].set_xlabel('t (d)', fontsize=14)
        ax[1].set_ylabel('N and P', fontsize=14)
        ax[1].legend(frameon=True)
        ax[1].axhline(y=0, color='k', linestyle='-', linewidth=1.0)
        
        # Set equal aspect ratios
        ax[0].set_box_aspect(0.7)
        ax[1].set_box_aspect(0.7)
        
        # Add timestamp
        fig.text(0.99, 0.01, 
                f'Generated: {datetime.now().strftime("%Y-%m-%d")}',
                fontsize=8, ha='right', alpha=0.5)
        
        # Adjust spacing between subplots
        plt.subplots_adjust(hspace=0.3)
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"Figure saved to {save_path}")
        
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
        fig, ax = simulate_biomass_models(
            t_max=args.t_max,
            S0=args.S0,
            k=args.k,
            save_path="fig2_2.png"
        )
        
        plt.show()
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
