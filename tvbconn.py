from tvb.simulator.lab import *
from tvb.simulator.plot.tools import *
from mpl_toolkits.mplot3d import Axes3D
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import SensorsEEG
from tvb.datatypes.projections import ProjectionMatrix, ProjectionSurfaceEEG
from tvb.simulator.models import WilsonCowan, Generic2dOscillator, ReducedWongWang, JansenRit
import time
import os

# zipped directory that contains connectivity/tractography info
master_path = 'C:/Users/Wayne Manselle/Documents/GitHub/tvb-data/tvb_data/berlinSubjects/DH_20120806/'

conn_fname = os.path.join(master_path, 'DH_20120806_Connectivity.zip')
conn = connectivity.Connectivity.from_file(conn_fname)

conn.configure()
print("connectivity loaded")
# print(conn.summary_info)

plot_connectivity(connectivity=conn)
# pyplot.show()

# triangulated cortical surface mesh
ctx_fname = os.path.join(master_path,'DH_20120806_Surface_Cortex.zip')
# maps nodes from cortex mesh to brain regions
region_fname = os.path.join(master_path,'DH_20120806_RegionMapping.txt')
# matlab matrix that describes how waves propagate to the surface
eeg_fname = os.path.join(master_path,'DH_20120806_ProjectionMatrix.mat')
ctx = cortex.Cortex.from_file(source_file=ctx_fname,
    region_mapping_file=region_fname,
    eeg_projection_file=eeg_fname)
ctx.configure()
print("cortex loaded")

pyplot.figure()
ax = pyplot.subplot(111, projection='3d')
x,y,z = ctx.vertices.T
ax.plot_trisurf(x, y, z, triangles=ctx.triangles, alpha=0.1, edgecolor='k')
# pyplot.show()
print("cortex plot ready")

# unit vectors that describe the location of eeg sensors
sensoreeg_fname = os.path.join(master_path,'DH_20120806_EEGLocations.txt')

rm = RegionMapping.from_file(region_fname)
sensorsEEG = SensorsEEG.from_file(sensoreeg_fname)
prEEG = ProjectionSurfaceEEG.from_file(eeg_fname)

fsamp = 1e3/1024.0 # 1024 Hz
mon = monitors.EEG(sensors=sensorsEEG, projection=prEEG,
    region_mapping=rm, period=fsamp)

sim = simulator.Simulator(
    connectivity=conn,
    # conduction speed: 3 mm/ms
    # coupling: linear - rescales activity propagated
    # stimulus: None - can be a spatiotemporal function

    # model: Generic 2D Oscillator - neural mass has two state variables that
    # represent a neuron's membrane potential and recovery; see the
    # mathematics paper for default values and equations; runtime 8016 s
    model=JansenRit(),
    # model: Wilson & Cowan - two neural masses have an excitatory/inhibitory
    # relationship; see paper for details; runtime 16031 s
    # model: Wong & Wang - reduced system of two non-linear coupled differential
    # equations based on the attractor network model; see paper; runtime 8177 s
    # model: Jansen & Rit - biologically inspired mathematical framework
    # originally conceived to simulate the spontaneous electrical activity of
    # neuronal assemblies, with a particular focus on alpha activity; had
    # some weird numpy overflow errors, only generated ~200 ms data; runtime 5929 s

    # integrator: Heun Deterministic
    # initial conditions: None
    monitors=mon,
    surface=ctx,
    simulation_length=1000.0 # ms
).configure()
print("sim configured")

print("sim running")
start = time.time()
result = sim.run()[0]
end = time.time()
print("sim done, took " + str(end-start) + " seconds")
pyplot.figure()
time, data = result
pyplot.plot(time, data[:, 0, :, 0], 'k', alpha=0.1)
pyplot.show()
