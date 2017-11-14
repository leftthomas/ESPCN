import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from tqdm import tqdm

from data_utils import DatasetFromFolder
from model import Net
from psnrmeter import PSNRMeter

UPSCALE_FACTOR = 3
NUM_EPOCHS = 200


def processor(sample):
    data, target, training = sample
    data = Variable(data)
    target = Variable(target)
    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()

    output = model(data)
    loss = criterion(output, target)

    return loss, output


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_psnr.reset()
    meter_loss.reset()


def on_forward(state):
    meter_psnr.add(state['output'].data, state['sample'][1])
    meter_loss.add(state['loss'].data[0])


def on_start_epoch(state):
    reset_meters()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[Epoch %d] Train Loss: %.4f (PSNR: %.2f)' % (
        state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_psnr_logger.log(state['epoch'], meter_psnr.value())

    reset_meters()

    engine.test(processor, val_loader)
    val_loss_logger.log(state['epoch'], meter_loss.value()[0])
    val_psnr_logger.log(state['epoch'], meter_psnr.value())

    print('[Epoch %d] Val Loss: %.4f (PSNR: %.2f)' % (
        state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % state['epoch'])


if __name__ == "__main__":

    train_set = DatasetFromFolder('data/train', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor())
    val_set = DatasetFromFolder('data/val', upscale_factor=UPSCALE_FACTOR, input_transform=transforms.ToTensor(),
                                target_transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=16, shuffle=False)

    model = Net(upscale_factor=UPSCALE_FACTOR)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_psnr = PSNRMeter()

    train_loss_logger = VisdomPlotLogger('line', opts={'title': 'Train Loss'})
    train_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Train PSNR'})
    val_loss_logger = VisdomPlotLogger('line', opts={'title': 'Val Loss'})
    val_psnr_logger = VisdomPlotLogger('line', opts={'title': 'Val PSNR'})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)
