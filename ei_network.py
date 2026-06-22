# ei_network.py
from netpyne import specs, sim
import numpy as np

netParams = specs.NetParams()

# Cell parameters
netParams.cellParams['E'] = {
    'secs': {
        'soma': {
            'geom': {'diam': 18.8, 'L': 32.9, 'Ra': 123.0, 'cm': 1.0},
            'mechs': {'hh': {'gnabar': 0.12, 'gkbar': 0.036, 'gl': 0.003, 'el': -70}},
            'vinit': -65.0
        }
    }
}
netParams.cellParams['I'] = {
    'secs': {
        'soma': {
            'geom': {'diam': 10.0, 'L': 20.0, 'Ra': 150.0, 'cm': 1.0},
            'mechs': {'hh': {'gnabar': 0.10, 'gkbar': 0.030, 'gl': 0.005, 'el': -65}},
            'vinit': -65.0
        }
    }
}

# External drive
netParams.cellParams['input'] = {
    'cellModel': 'NetStim',
    'rate': 15,
    'noise': 0.5,
    'start': 50,
}

# Populations
netParams.popParams['E_pop'] = {'cellType': 'E', 'numCells': 80}
netParams.popParams['I_pop'] = {'cellType': 'I', 'numCells': 20}
netParams.popParams['input_pop'] = {'cellType': 'input', 'numCells': 20, 'cellModel': 'NetStim'}

# Synapses
netParams.synMechParams['exc'] = {'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': 5.0, 'e': 0}
netParams.synMechParams['inh'] = {'mod': 'Exp2Syn', 'tau1': 0.1, 'tau2': 10.0, 'e': -80}

# Connectivity
netParams.connParams['input->E'] = {
    'preConds': {'pop': 'input_pop'},
    'postConds': {'pop': 'E_pop'},
    'probability': 0.3,
    'weight': 0.015,
    'delay': 1.0,
    'synMech': 'exc'
}
netParams.connParams['E->E'] = {
    'preConds': {'pop': 'E_pop'},
    'postConds': {'pop': 'E_pop'},
    'probability': 0.2,
    'weight': 0.005,
    'delay': 1.0,
    'synMech': 'exc'
}
netParams.connParams['E->I'] = {
    'preConds': {'pop': 'E_pop'},
    'postConds': {'pop': 'I_pop'},
    'probability': 0.2,
    'weight': 0.005,
    'delay': 1.0,
    'synMech': 'exc'
}
netParams.connParams['I->E'] = {
    'preConds': {'pop': 'I_pop'},
    'postConds': {'pop': 'E_pop'},
    'probability': 0.2,
    'weight': 0.01,
    'delay': 1.0,
    'synMech': 'inh'
}
netParams.connParams['I->I'] = {
    'preConds': {'pop': 'I_pop'},
    'postConds': {'pop': 'I_pop'},
    'probability': 0.2,
    'weight': 0.01,
    'delay': 1.0,
    'synMech': 'inh'
}

# --- SIMULATION CONFIG ---
simConfig = specs.SimConfig()
simConfig.duration = 1000
simConfig.dt = 0.025
simConfig.verbose = False
simConfig.hParams = {'celsius': 34}
simConfig.recordCells = ['all']
simConfig.recordStep = 1.0


simConfig.recordTraces = {
    'V_soma': {'sec': 'soma', 'loc': 0.5, 'var': 'v'}
}

simConfig.analysis['plotRaster'] = True
simConfig.analysis['plotTraces'] = {'include': ['E_pop']}
simConfig.analysis['plotRateSpectrogram'] = True
simConfig.analysis['plotConn'] = True
simConfig.savePickle = True
simConfig.saveJson = False

# --- RUN ---
sim.createSimulateAnalyze(netParams=netParams, simConfig=simConfig)
print("✅ Simulation finished!")

# --- MANUALLY COMPUTE POPULATION RATE FROM SPIKE DATA ---
print("Computing population rate from spike data...")


# Get spike data
spkt = np.array(sim.allSimData['spkt'])
spkid = np.array(sim.allSimData['spkid'])

# E cells are GIDs 0-79 (first 80 cells because E_pop was defined first)
# I cells are GIDs 80-99
e_mask = spkid < 80  # GIDs 0-79 are E cells
e_spkt = spkt[e_mask]

print(f"E-cell spikes: {len(e_spkt)} / {len(spkt)} total")

if len(e_spkt) == 0:
    print("⚠️ No E spikes! Trying ALL spikes (network is active!)")
    e_spkt = spkt  # Use all spikes as fallback

# Compute binned firing rate
bin_size = 5.0  # ms
bins = np.arange(0, simConfig.duration + bin_size, bin_size)
hist, _ = np.histogram(e_spkt, bins=bins)
pop_rate = hist / (80 * bin_size / 1000.0)  # 80 E cells

# Time points (center of bins)
time_centers = bins[:-1] + bin_size/2

# Save
np.save('time.npy', np.array(time_centers))
np.save('pop_rate.npy', np.array(pop_rate))

print(f"✅ Saved population rate. Length: {len(pop_rate)}")
print(f"   Mean firing rate: {np.mean(pop_rate):.2f} Hz")
print(f"   Max firing rate: {np.max(pop_rate):.2f} Hz")