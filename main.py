import argparse
from math import log10
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Net
from data_utils import get_training_set, get_test_set

# Train settings
parser = argparse.ArgumentParser(description='Train Super Resolution')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
opt = parser.parse_args()


print('===> Loading datasets')
train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=64, shuffle=False)


print('===> Building model')
model = Net(upscale_factor=opt.upscale_factor)
criterion = nn.MSELoss()

if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        data, target = Variable(batch[0]), Variable(batch[1])
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        loss = criterion(model(data), target)
        epoch_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def test():
    avg_psnr = 0
    for batch in testing_data_loader:
        data, target = Variable(batch[0]), Variable(batch[1])
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        prediction = model(data)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))


def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


for e in range(1, 3):
    train(e)
    test()
    checkpoint(e)
