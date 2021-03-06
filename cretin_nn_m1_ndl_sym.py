import numpy as np
import matplotlib.pyplot as plt
import pca_data_cretin_m1_ndl_sym as pca_data_cretin
import torch
import torch.nn as nn
from torch.utils import data as data_utils
from logger import Logger
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import shuffle
from tqdm import tqdm
import os


torch.set_printoptions(precision=3) #this doesn't seem to do anything

TENSORBOARD_DIR = '/usr/WS1/hammel1/proj/tensorboard/'
CHECKPOINT_DIR = '/usr/WS1/hammel1/proj/checkpoints/'
 
experiment_id = datetime.now().isoformat()

plt.ion()
plt.close('all')

USE_GPU = torch.cuda.is_available()

device = torch.device('cuda' if USE_GPU else 'cpu')
torch.set_default_tensor_type(
    'torch.cuda.FloatTensor' if USE_GPU else 'torch.FloatTensor'
)

hu, PC, xtrain_pca, xtest_pca, Xtrain, Xtest, Ytrain, Ytest, X, Y = pca_data_cretin.pca_data()

trainer = data_utils.TensorDataset(torch.from_numpy(xtrain_pca).float().to(device), torch.from_numpy(Ytrain).float().to(device))
tester = data_utils.TensorDataset(torch.from_numpy(xtest_pca).float().to(device), torch.from_numpy(Ytest).float().to(device))

np.save(f"/usr/WS1/hammel1/proj/data/{experiment_id}", [xtrain_pca, Ytrain, xtest_pca, Ytest])

# print(f"Input Dimension D_in_1 will be: {PC}")
N, PC = xtrain_pca.shape

BATCH_SZ, D_in_1, L, M, Q, R, D_out = 2048, PC, 2000, 1000, 500, 50, 11
EPOCHS = 20_000


if __name__ == '__main__':
    model = torch.nn.Sequential(
        torch.nn.BatchNorm1d(D_in_1),
        torch.nn.Linear(D_in_1, L),
        torch.nn.Dropout(0.01),
        torch.nn.Tanh(),
        torch.nn.Linear(L, M),
        torch.nn.Dropout(0.01),
        torch.nn.Tanh(),
        torch.nn.Linear(M, Q),
        torch.nn.Dropout(0.01),
        torch.nn.Tanh(),
        torch.nn.Linear(Q, R),
        #torch.nn.Dropout(0.01),
        torch.nn.Tanh(),
        torch.nn.Linear(R, D_out),
    )

    if USE_GPU:
        print("="*80)
        print("Model is using GPU")
        print("="*80)
        model.cuda()
    else:
        print("="*80)
        print("Model is using CPU")
        print("="*80)

    loss_fn = torch.nn.L1Loss()
    #loss_fn = torch.nn.MSELoss()
    best_loss = float('inf')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, min_lr=1e-10, patience=500, factor=0.5, verbose=True,
    )

    print('Logging experiment as: ', experiment_id)

    logger = Logger(os.path.join(TENSORBOARD_DIR, experiment_id))

    # Load data
    train_loader = torch.utils.data.DataLoader(trainer, batch_size=BATCH_SZ, shuffle=False)
    test_loader = torch.utils.data.DataLoader(tester, batch_size=BATCH_SZ, shuffle=False)

    pbar = tqdm(range(EPOCHS))

    for epoch in pbar:

        train_losses = []
        model.train()

        for x_batch, y_batch in train_loader:

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        model.eval()  # tell pytorch the model is in test mode


        for x_batch, y_batch in test_loader:

            y_pred = model(x_batch)
            test_loss = loss_fn(y_pred, y_batch)

            lr_scheduler.step(test_loss)    # I think this actually needs to be outside the loop, but will have to calculate the average

        test_loss = test_loss.item()

        pbar.set_description(
            f"mean train loss: {train_loss:.4f}, mean test loss: {test_loss:.4f}"
        )

        logger.scalar_summary('lr', optimizer.param_groups[0]['lr'], epoch)
        logger.scalar_summary('train loss', train_loss, epoch)
        logger.scalar_summary('test loss', test_loss, epoch)

        # 2. Log values and gradients of the parameters (histogram summary)
        # curious to see if this works!
        # for tag, value in model.named_parameters():
        #     tag = tag.replace('.', '/')
        #     logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
            # logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)

        if best_loss < test_loss:
            torch.save(model, os.path.join(CHECKPOINT_DIR, f'{experiment_id}_best_val'))
            best_loss = test_loss

        torch.save(model, os.path.join(CHECKPOINT_DIR, f'{experiment_id}_latest'))

    I_ = model(torch.from_numpy(xtest_pca).float().to(device))

    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.scatter(Ytrain[::1,3], Xtrain.mean(1)[::1], c='g')
    plt.scatter(Ytest[::1,3],Xtest.mean(1)[::1],  c='r')
    plt.scatter(I_.cpu()[::1,3].detach().squeeze(-1), Xtest.mean(1)[::1],  c='m')
    plt.ylabel('Int (Xtest.mean(1))')
    plt.xlabel('mix (Ytest[:,3])') 
    
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.scatter(Ytrain[::1,4], Xtrain.mean(1)[::1], c='g')
    plt.scatter(Ytest[::1,4],Xtest.mean(1)[::1],  c='r')
    plt.scatter(I_.cpu()[::1,4].detach().squeeze(-1), Xtest.mean(1)[::1],  c='m')
    plt.ylabel('Int (Xtest.mean(1)')
    plt.xlabel('mixhot (Ytest[:,4])')
 
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.scatter(Ytrain[::1,7], Xtrain.mean(1)[::1], c='g')
    plt.scatter(Ytest[::1,7],Xtest.mean(1)[::1],  c='r')
    plt.scatter(I_.cpu()[::1,7].detach().squeeze(-1), Xtest.mean(1)[::1],  c='m')
    plt.ylabel('Int (Xtest.mean(1)')
    plt.xlabel('t3 (Ytest[:,7])')
 
    fig = plt.figure(dpi=100, figsize=(5, 4))
    plt.scatter(Ytrain[::1,9], Xtrain.mean(1)[::1], c='g')
    plt.scatter(Ytest[::1,9],Xtest.mean(1)[::1],  c='r')
    plt.scatter(I_.cpu()[::1,9].detach().squeeze(-1), Xtest.mean(1)[::1],  c='m')
    plt.ylabel('Int (Xtest.mean(1)')
    plt.xlabel('mch (Ytest[:,9])')
    

    
    fig = plt.figure(dpi=100, figsize=(5, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Ytest[::1,3], Ytest[::1,4], Ytest[::1,9], c='r', marker='o')
    ax.scatter(Ytrain[::1,3], Ytrain[::1,4], Ytrain[::1,9], c='g', marker='o')
    ax.scatter(I_.cpu()[::1,3].detach().squeeze(-1), I_.cpu()[::1,4].detach().squeeze(-1), I_.cpu()[::1,9].detach().squeeze(-1), c='m', marker='o') 
    ax.set_xlabel('mu (Ytest[:,0])')
    ax.set_ylabel('std (Ytest[:,1])')
    ax.set_zlabel('amp (Ytest[:,2])')
