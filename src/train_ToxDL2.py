import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import pytorch_model_store, data_loader_result_write, evaluation_result_obtained
from parameters.test_000 import DATA, model_best_save_path, test_predicted_save_path, independent_predicted_save_path
from parameters.test_000 import learning_rate, epochs, device, gamma, step_size
from dataset import ToxProteinDataModule
from model import ToxDL_GCN_Network


def run_train():
    # load the data, checkpoints, and define the loss function
    dataset = ToxProteinDataModule(DATA)
    train_loader = dataset.train_dataloader()
    valid_loader = dataset.valid_dataloader()
    test_loader = dataset.test_dataloader()
    independent_loader = dataset.independent_dataloader()

    net = ToxDL_GCN_Network().to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = nn.BCELoss().to(device)

    valid_loss_min = np.Inf
    for epoch in range(epochs):
        net.train()
        scheduler.step()
        # perform the actual training
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            data_copy = data.clone().detach()
            # compute the models predictions and the loss
            pred = net.forward(data_copy).squeeze()
            label = data.y.squeeze().to(torch.float32)
            loss = loss_fn(pred, label)

            # perform one step of backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {0} [{1}/{2} ({3:.0f}%)]\tLoss: {4:.6f}\tLearning Rate: {5}'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), scheduler.get_last_lr()[0]))

        valid_loss = 0
        net.eval()
        for data in valid_loader:
            data = data.to(device)
            data_copy = data.clone().detach()
            pred = net.forward(data_copy).squeeze().to(torch.float32)
            label = data.y.squeeze().to(torch.float32)
            loss = loss_fn(pred, label)
            valid_loss += loss.item()

        valid_loss /= len(valid_loader.dataset)
        print('VAL set: Average loss: {0:.4f}'.format(valid_loss))
        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            pytorch_model_store(net, model_best_save_path)
            data_loader_result_write(model_best_save_path, test_loader, test_predicted_save_path)
            data_loader_result_write(model_best_save_path, independent_loader, independent_predicted_save_path)
    evaluation_result_obtained(test_predicted_save_path)
    evaluation_result_obtained(independent_predicted_save_path)


if __name__ == '__main__':
    run_train()
