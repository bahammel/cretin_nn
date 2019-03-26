import pca_data_cretin_m1_ndl_sym as pca_data_cretin
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

USE_GPU = torch.cuda.is_available()
# All with BATCH_SZ, D_in_1, L, M, Q, R, D_out = 512, PC, 5000, 1000, 500, 100, 11
#exp_id = '2ax.xaxis.set_scale('log')ax.xaxis.set_scale('log')019-01-22T14:06:58.448284'
#exp_id = '2019-01-24T11:58:56.806565'  # MSE Loss, 0.1 DropOut all but last layer
#exp_id = '2019-01-24T12:40:40.486694'  # L1 Loss, 0.1 DropOut all but last layer
#exp_id = '2019-01-24T13:22:40.175860'   # MSE Loss, 0.2 DropOut all layers
#exp_id = '2019-01-24T13:54:41.011544'   #L1 Loss, 0.2 DropOut all layers

### ************ Best Run*************
#exp_id = '2019-01-24T14:44:38.155622'   #L1 Loss, Renormalized so most inputs are between 0-1, 0.01 DropOut all but last layer
# All with BATCH_SZ, D_in_1, L, M, Q, R, D_out = 512, PC, 1000, 500, 100, 50, 11
#exp_id = '2019-01-24T17:24:05.194929'   #Same as above
#exp_id = '2019-01-24T17:42:26.360158'   #Repeat
#exp_id = '2019-01-24T19:04:57.708293'   #Renomalized so ALL inputs are between 0-1
#exp_id = '2019-01-28T18:05:35.953416'    #~25k runs, D_out = 2048, PC, 2000, 1000, 500, 50, 11, 20k epochs, NO noise, pca 20-ish ? (99.9%) 
#exp_id = '2019-01-29T14:24:15.226733'    #~25k runs, D_out = 2048, PC, 2000, 1000, 500, 50, 11, 20k epochs, pct_noise=50,pca~135 (99.9%)
#exp_id = '2019-01-29T17:12:22.276173'    #~25k runs, D_out = 2048, PC, 2000, 1000, 500, 50, 11, 20k epochs, pct_noise=20,pca=95 (99.9%)
#exp_id = '2019-01-30T14:20:25.849339'    #~25k runs, D_out = 2048, PC, 2000, 1000, 500, 50, 11, 20k epochs, pct_noise=10,pca=48 (99.9%)
#exp_id = '2019-01-30T16:32:50.954897'    #~25k runs, D_out = 2048, PC, 2000, 1000, 500, 50, 11, 20k epochs, pct_noise=10,pca=20 (forced)
#exp_id = '2019-01-31T10:27:30.655803'     #~25k runs, D_out = 2048, PC, 2000, 1000, 500, 50, 11, 10k epochs, pct_noise=10,pca=30 (forced)
#exp_id = '2019-02-05T10:09:35.774687'     #~25k runs, D_out = 2048, PC, 2000, 1000, 500, 50, 11, 10k epochs, pct_noise=20,pca=30 (forced)
#exp_id = '2019-02-13T11:55:41.255264'      #Test new dataloader ~25k runs, D_out = 2048, PC, 2000, 1000, 500, 50, 11, 5k epochs, pct_noise=20,pca=30 (forced)

#exp_id = '2019-02-14T10:54:20.636256'    #New dataloader ~25k runs, D_out = 2048, PC, 2000, 1000, 500, 50, 11, 20k epochs, pct_noise=20,pca=30 (forced)
exp_id = '2019-03-22T16:25:13.104017'    #New dataloader ~25k runs, D_out = 2048, PC, 2000, 1000, 500, 50, 11, 20k epochs, pct_noise=20,pca=30 (forced)
device = torch.device('cuda' if USE_GPU else 'cpu')
torch.set_default_tensor_type(
    'torch.cuda.FloatTensor' if USE_GPU else 'torch.FloatTensor'
)

CHECKPOINT_DIR = f'/usr/WS1/hammel1/proj/checkpoints/{exp_id}_latest'
print('exp_id is:', exp_id)
print('CHECKPOINT_DIR is:', CHECKPOINT_DIR)

xtrain_pca, ytrain, xtest_pca, ytest = np.load(f'/usr/WS1/hammel1/proj//data/{exp_id}.npy')

Xtrain = torch.Tensor(xtrain_pca)
Xtest = torch.Tensor(xtest_pca)
Ytrain = torch.Tensor(ytrain)
Ytest = torch.Tensor(ytest)

model = torch.load(CHECKPOINT_DIR)

if USE_GPU:
    print("="*80)
    print("Model is using GPU")
    print("="*80)
    model.cuda()

model.eval()

# Ben's original
#y = ytrain
#y_pred = model(Xtrain).cpu().detach().numpy()

# Do this to see error in fiting Training set
y = Ytrain
y_pred = model(Xtrain)
ranges = ytrain.max(axis=0) - ytrain.min(axis=0)

#mse = np.mean((y_pred - y)**2, axis=0)
mse = ((y_pred - y)**2).mean(0)
mse = mse.cpu().detach().numpy()
pct_err = mse / y.mean(0).cpu().detach().numpy() * 100

# Do this to see error in fiting Test set
y_test = Ytest
y_pred_test = model(Xtest)
ranges_test = ytest.max(axis=0) - ytest.min(axis=0)

#mse = np.mean((y_pred - y)**2, axis=0)
mse_test = ((y_pred - y)**2).mean(0)
mse_test = mse_test.cpu().detach().numpy()
#pct_err_test = mse_test / ranges_test * 100
pct_err_test = mse_test / y_test.mean(0).cpu().detach().numpy() * 100

# pct_err = mse / ytrain.mean(0) * 100

for i, inp in enumerate(['t1', 'n1', 'mhot', 'mix', 'mix_hot', 'theta', 'n2', 't3', 'n3', 'mch', 'rmax']):
    #print(f"{inp:.<30}{pct_err[i]:.2f}, {pct_err_test[i]:.2f}")
    print(f"{inp:.<30}{pct_err[i]:.3e}, {pct_err_test[i]:.3e}")

print('Quadrature sum of training errors', np.sqrt(sum(pct_err**2)))
print('Quadrature sum of test errors', np.sqrt(sum(pct_err_test**2)))

fig = plt.figure(dpi=100, figsize=(5, 4))
#plt.scatter(Ytrain.cpu()[::10,3], Xtrain.cpu().mean(1)[::10], c='g')
plt.scatter(Ytest.cpu()[::10,3],(0.1 + Xtest.cpu().mean(1)[::10]),  c='r')
plt.scatter(y_pred_test.cpu()[::10,3].detach().squeeze(-1), (0.1 + Xtest.cpu().mean(1)[::10]),  c='m')
plt.xlim(1.e-3, 1.0)
plt.ylim(2.e-3, 2.0)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Int (Xtest.mean(1))')
plt.xlabel('mix (Ytest[:,3])') 

fig = plt.figure(dpi=100, figsize=(5, 4))
#plt.scatter(Ytrain.cpu()[::10,4], Xtrain.cpu().mean(1)[::10], c='g')
plt.scatter(Ytest.cpu()[::10,4], (0.1 + Xtest.cpu().mean(1)[::10]),  c='r')
plt.scatter(y_pred_test.cpu()[::10,4].detach().squeeze(-1), (0.1 + Xtest.cpu().mean(1)[::10]),  c='m')
plt.xlim(1.e-3, 1.0)
plt.ylim(2.e-3, 2.0)
plt.xscale('log')
plt.yscale('log')
plt.ylabel('Int (Xtest.mean(1)')
plt.xlabel('mixhot (Ytest[:,4])')

fig = plt.figure(dpi=100, figsize=(5, 4))
#plt.scatter(Ytrain.cpu()[::10,7], Xtrain.cpu().mean(1)[::10], c='g')
plt.scatter(Ytest.cpu()[::10,7],(0.1 + Xtest.cpu().mean(1)[::10]),  c='r')
plt.scatter(y_pred_test.cpu()[::10,7].detach().squeeze(-1), (0.1 + Xtest.cpu().mean(1)[::10]),  c='m')
plt.ylim(2.e-3, 2.0)
plt.yscale('log')
plt.ylabel('Int (Xtest.mean(1)')
plt.xlabel('t3 (Ytest[:,7])')

fig = plt.figure(dpi=100, figsize=(5, 4))
#plt.scatter(Ytrain.cpu()[::10,9], Xtrain.cpu().mean(1)[::10], c='g')
plt.scatter(Ytest.cpu()[::10,9],(0.1 + Xtest.cpu().mean(1)[::10]),  c='r')
plt.scatter(y_pred_test.cpu()[::10,9].detach().squeeze(-1), (0.1 + Xtest.cpu().mean(1)[::10]),  c='m')
plt.ylim(2.e-3, 2.0)
plt.yscale('log')
plt.ylabel('Int (Xtest.mean(1)')
plt.xlabel('mch (Ytest[:,9])')

fig = plt.figure(dpi=100, figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.log(Ytest.cpu()[::2,3]), np.log(Ytest.cpu()[::2,4]), np.log(Ytest.cpu()[::2,9]), c='r', marker='o')
#ax.scatter(Ytrain.cpu()[::20,3], Ytrain.cpu()[::20,4], Ytrain.cpu()[::20,9], c='g', marker='o')
ax.scatter(np.log(y_pred_test.cpu()[::2,3].detach().squeeze(-1)), np.log(y_pred_test.cpu()[::2,4].detach().squeeze(-1)), np.log(y_pred_test.cpu()[::2,9].detach().squeeze(-1)), c='m', marker='o') 
plt.xlim(np.log(1.e-3), np.log(1.1))
plt.ylim(np.log(1.e-3), np.log(1.1))
#plt.ylim(1.e-3, 2.0)
#ax.xaxis.set_scale('log')
#ax.yaxis.set_scale('log')
#plt.xscale('log')
#plt.yscale('log')
ax.set_xlabel('log(mix) (Ytest[:,3])')
ax.set_ylabel('log(mixhot) (Ytest[:,4])')
ax.set_zlabel('mch (Ytest[:,9])')

