import numpy as np

data = np.load("/home/seohy/colcon_ws/src/olaf/ffw/code/dataGet/delta_tau_dataset.npz")

print(data.files)

print(data['q'].shape)
print(data['qdot'].shape)
print(data['tau_mpc'].shape)
print(data['delta_tau'].shape)
