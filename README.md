# System Identification of Seizure Dynamics in a Balanced E/I Network

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NetPyNE](https://img.shields.io/badge/NetPyNE-1.0.0-green.svg)](https://www.netpyne.org/)
[![NEURON](https://img.shields.io/badge/NEURON-8.0-orange.svg)](https://neuron.yale.edu/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Overview

A complete computational neuroscience pipeline to detect seizure-like activity in a balanced excitatory-inhibitory network using:

- **Biophysical modeling** (NetPyNE/NEURON)
- **System identification** (SINDy)
- **Clinical validation** (Lyapunov exponents)

The pipeline discovers governing ODEs from neural data and classifies the brain state as chaotic (healthy), periodic (seizure), or stable (anesthesia).

## 🚀 Quick Start


# Install dependencies
netpyne>=1.0.0
neuron>=8.0.0
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.4.0
pysindy>=1.7.0
nolds>=0.5.2
scikit-learn>=1.0.0


# Run the full pipeline
python ei_network.py          # Simulate network & save population rate
python sindy_fit_ei.py        # Discover ODEs using SINDy
python lyapunov_validation.py # Validate with Lyapunov exponents


#Discovered ODEs (4-dimensional system)
x0' = -0.154 x0 - 1.774 x1 + 2.072 x2 - 0.146 x3 
    + 0.902 x0² - 5.714 x0x1 + 6.499 x0x2 - 3.241 x0x3
    + 4.215 x1² - 8.631 x1x2 + 7.099 x1x3
    + 3.155 x2² - 4.614 x2x3 + 0.322 x3²

x1' = 0.315 x0 - 2.423 x1 + 2.107 x2 - 0.001 x3
    + 1.191 x0² - 5.818 x0x1 + 7.229 x0x2 - 3.885 x0x3
    + 4.702 x1² - 11.015 x1x2 + 8.178 x1x3
    + 3.734 x2² - 4.189 x2x3 - 0.125 x3²

x2' = -0.049 x0 + 5.397 x1 - 5.104 x2 - 0.243 x3
    - 0.761 x0² + 9.409 x0x1 - 13.236 x0x2 + 5.099 x0x3
    - 5.429 x1² + 15.109 x1x2 - 12.652 x1x3
    - 4.859 x2² + 7.234 x2x3 + 0.068 x3²

x3' = 0.034 x0 - 0.833 x1 + 1.125 x2 - 0.319 x3
    + 0.098 x0² - 1.739 x0x1 + 2.449 x0x2 - 1.117 x0x3
    + 1.263 x1² - 3.554 x1x2 + 2.648 x1x3
    + 3.092 x2² - 4.780 x2x3 + 1.635 x3²

#provided by 8navid@gmail.com
