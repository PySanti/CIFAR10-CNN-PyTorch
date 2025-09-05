import torch
import numpy as np
import time
import torchvision.transforms as transforms
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from utils.MACROS import NUM_WORKERS, BATCH_SIZE
from utils.PlainCNN import PlainCNN
from utils.plot_performance import plot_performance

if __name__ == "__main__":

    transform_train = transforms.Compose([
        # data augmentation
        transforms.RandomHorizontalFlip(),         # Volteo horizontal aleatorio
        transforms.RandomRotation(15),             # Rotación aleatoria ±15°
        transforms.RandomCrop(32, padding=4),      # Recorte aleatorio con padding
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Variación de color
        transforms.ToTensor(),                        
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])


    trainset = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size

    torch.manual_seed(42)
    trainset, valset = random_split(trainset, [train_size, val_size])
    valset.dataset.transform = transform_test

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, persistent_workers=True)
    valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, persistent_workers=True)

    cnn = PlainCNN(3).to('cuda')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=cnn.parameters(), lr=1e-2, weight_decay=1e-4)
    epoch_train_loss = []
    epoch_val_loss = []

    for i in range(100):

        train_prec = []
        val_prec = []
        train_loss = []
        val_loss = []

        t1 = time.time()
        cnn.train()

        if (i+1)%15 == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']*0.1
            print(f"Valor actual del lr : {optimizer.param_groups[0]['lr']}")

        for a, (X_batch, Y_batch) in enumerate(trainloader):
            X_batch, Y_batch = X_batch.to('cuda'), Y_batch.to('cuda')

            optimizer.zero_grad()
            output = cnn(X_batch)
            loss = criterion(output, Y_batch)

            loss.backward()
            optimizer.step()


            # metrics
            _, pred = torch.max(output, 1)
            train_prec.append((pred == Y_batch).cpu().sum() / len(X_batch))
            train_loss.append(loss.item())


        cnn.eval()
        with torch.no_grad():
            for a, (X_batch, Y_batch) in enumerate(valloader):
                X_batch, Y_batch = X_batch.to('cuda'), Y_batch.to('cuda')

                output = cnn(X_batch)
                loss = criterion(output, Y_batch)

                # metrics
                _, pred = torch.max(output, 1)
                val_prec.append((pred == Y_batch).cpu().sum() / len(X_batch))
                val_loss.append(loss.item())



        print(f"""
              Epoch : {i}

                    Train Loss : {np.mean(train_loss):.3f}
                    Train prec : {np.mean(train_prec):.3f}

                    Val loss : {np.mean(val_loss):.3f}
                    Val prec : {np.mean(val_prec):.3f}

                    Time : {time.time()-t1}
              """)
        epoch_train_loss.append(np.mean(train_loss))
        epoch_val_loss.append(np.mean(val_loss))

    plot_performance(epoch_train_loss, epoch_val_loss)
