import time
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.io import loadmat

from dictlearn import ak_svd_update
from contrastive_dl_toolbox import extract_data, optimize_dictionary, atom_distribution


# Set seeds for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

# Load the configuration file
with open('config.json', 'r') as f:
    config = json.load(f)

# Access dataset-specific configuration
dataset_name = 'yaleb'
dataset_config = config['datasets'][dataset_name]

# Load dataset
data = loadmat('mats/' + dataset_config['data_path'])
X_train, X_test, y_train, y_test = extract_data(data['featureMat'], data['labelMat'], dataset_config['number'])
device = 'cpu'

# Convert to PyTorch tensors and move to the specified device
X_train = torch.tensor(X_train, dtype=torch.float64).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)

# Dataset properties
n_classes = len(y_train.unique())
n_features, n_samples = X_train.shape

n_atoms_per_class = config['n_components']
n_atoms = n_classes * n_atoms_per_class

# Initialize and normalize dictionary
D = torch.randn(n_features, n_atoms, dtype=X_train.dtype)
# normalize dictionary atoms
D = torch.nn.functional.normalize(D, dim=0)
D.requires_grad_(True)
D0 = D.clone().detach()

# SimCLR parameters
batch_size = config['batch_size']
d_up_iterations = config['d_up_iterations']
learning_rate = config['learning_rate']
max_angle = config['max_angle']
cutout_perc = config['cutout_perc']
noise_std = config['noise_std']
blur_std = config['blur_std']
transformations = ['rotate', 'cutout', 'gaussian_noise', 'gaussian_blur']

# DL parameters
n_iterations = config['n_iterations']
n_nonzero_coefs = config['n_nonzero_coefs']

# optimize dictionary following the SimCLR framework
t0 = time.time()
D = optimize_dictionary(D, X_train, n_nonzero_coefs, batch_size, d_up_iterations, learning_rate, transformations, max_angle, cutout_perc, noise_std, blur_std)
tf = time.time()
opt_dict_time = tf - t0

# atoms assignment to classes
atom_classes = atom_distribution(X_train, D, n_nonzero_coefs, n_classes)

# Compare the two DL problems
Y = X_train
D.requires_grad = False

# trained dictionary
errs = []

t0 = time.time()
for idx_iter in tqdm(range(n_iterations)):
    D, X = ak_svd_update(Y, D, n_nonzero_coefs)
    errs.append(torch.linalg.norm(Y - D.mm(X), ord=2).item() / (Y.shape[0]*Y.shape[1]))
tf = time.time()    
ak_svd_time = tf - t0 


# origin dictionary
errs0 = []
n_additional_iterations = round(opt_dict_time / (ak_svd_time / n_iterations))

for idx_iter in tqdm(range(n_iterations + n_additional_iterations)):
    D0, X = ak_svd_update(Y, D0, n_nonzero_coefs)
    errs0.append(torch.linalg.norm(Y - D0.mm(X), ord=2).item() / (Y.shape[0]*Y.shape[1]))
 

plt.plot(list(range(1, n_iterations + n_additional_iterations + 1)), errs0, label='AK-SVD')
plt.plot(list(range(1, n_iterations + 1)), errs, label='SimCDL + AK-SVD')
plt.legend()
plt.grid()
plt.title('Representation error')
plt.xlabel('iter')
plt.ylabel('error')

print()
print(f'errs: {errs[-1]}')
print(f'errs0: {errs0[-1]}')
print(f'delta: {errs0[-1] - errs[-1]}')

