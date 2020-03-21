from voice_dataset import VoiceData
import torch
from torch.utils.data import DataLoader
from conv_model import ConvMod
import torch.nn as nn
from torch.nn import DataParallel
import time


def train_conv(epochs, batch_size, dev_ids, learning_rate=0.001, save_file=None, show_batch=True):

    # get data and dataloader
    voice_data = VoiceData()
    dataloader = DataLoader(voice_data,
                            shuffle=True,
                            batch_size=batch_size)

    # get device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # declare model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = ConvMod().to(device)
        model = DataParallel(model, device_ids=dev_ids)
    else:
        model = ConvMod() 

    # training mode
    model.train()

    # declare training methodology
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    track_epoch_time = []
    for e in range(epochs):
        start = time.time()
        print("Starting Epoch", e)
        # Loop over minibatches
        for i, (x, y) in enumerate(dataloader):

            # move to device
            x = x.unsqueeze(1)
            x = x.to(device)
            y = y.to(device)

            # zero gradient
            optimizer.zero_grad()

            # forward
            y_ = model(x)
            loss = criterion(y_,y)
            loss.backward()
            optimizer.step() 

            if show_batch:
                print("batch {}/{} \t loss: {}".format(i, len(voice_data)/batch_size, float(loss)), end='\r')

        epoch_time = time.time()-start
        print("\nEpoch-time: ", epoch_time)
        track_epoch_time.append(epoch_time)

        # save file
        if save_file: 
            torch.save(model, save_file)

    track_epoch_time = track_epoch_time[1:]
    total_epoch_time = sum(track_epoch_time)
    avg_epoch_time = total_epoch_time/len(track_epoch_time)

    return({'GPUs':len(dev_ids), 'batch_size':batch_size, 'epoch_time':avg_epoch_time})

if __name__ == '__main__':
    epochs = 5
    batch_size = 200
    dev_ids = [0]
    learning_rate = 0.0005

    out = train_conv(epochs, batch_size, dev_ids, learning_rate=learning_rate)
    print(out)
