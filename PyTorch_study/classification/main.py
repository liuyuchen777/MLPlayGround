import DefNN
import torch.optim as optim
import torch.nn as nn
import torch
import GatherData
import torch
import ShowImg
import torchvision

def main():
    # define network
    net = DefNN.Net()
    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # train the network
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(GatherData.trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training')
    # save model
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    # test process
    dataiter = iter(GatherData.testloader)
    images, labels = dataiter.next()
    # print images
    ShowImg.imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % GatherData.classes[labels[j]] for j in range(4)))
    # predict
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % GatherData.classes[predicted[j]] for j in range(4)))



if __name__ == "__main__":
    main()