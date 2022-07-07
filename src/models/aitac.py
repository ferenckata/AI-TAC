"""Main AITAC model"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import copy

# Convolutional neural network
class AITAC(nn.Module):
    """Main AITAC model
    """
    def __init__(self, num_classes, num_filters):
        """Main AITAC model
        """
        super().__init__()
        # for layer one, separate convolution and relu step from maxpool and batch normalization
        # to extract convolutional filters
        self.layer1_conv = nn.Sequential(
            # padding is done in forward method along 1 dimension only, Conv2D would do in both dimensions
            nn.Conv2d(in_channels=1,
                      out_channels=num_filters,
                      kernel_size=(4, 19),
                      stride=1,
                      padding=0),
            nn.ReLU())

        self.layer1_process = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3), padding=(0,1)),
            nn.BatchNorm2d(num_filters))

        self.layer2 = nn.Sequential(
            # padding is done in forward method along 1 dimension only, Conv2D would do in both dimensions
            nn.Conv2d(in_channels=num_filters,
                      out_channels=200,
                      kernel_size=(1, 11),
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,4), stride=(1,4), padding=(0,1)),
            nn.BatchNorm2d(200))

        self.layer3 = nn.Sequential(
            # padding is done in forward method along 1 dimension only, Conv2D would do in both dimensions
            nn.Conv2d(in_channels=200,
                      out_channels=200,
                      kernel_size=(1, 7),
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4), stride=(1,4), padding=(0,1)),
            nn.BatchNorm2d(200))

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=1000,
                      out_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.03))

        self.layer5 = nn.Sequential(
            nn.Linear(in_features=1000,
                      out_features=1000),
            nn.ReLU(),
            nn.Dropout(p=0.03))

        self.layer6 = nn.Sequential(
                nn.Linear(in_features=1000,
                          out_features=num_classes))#,
                #nn.Sigmoid())


    def forward(self, data_in):
        """Forward pass
        Parameters
        ----------
        data_in: np.array
            Input data to train on dimensions?
        """
        # run all layers on input data
        # add dummy dimension to input (for num channels=1)
        data_in = torch.unsqueeze(data_in, 1)

        # Run convolutional layers
        # padding - last dimension goes first, done here so that it is added along one dimension only
        data_in = F.pad(data_in, (9, 9), mode='constant', value=0)
        out = self.layer1_conv(data_in)
        activations = torch.squeeze(out)
        out = self.layer1_process(out)
        
        out = F.pad(out, (5, 5), mode='constant', value=0)
        out = self.layer2(out)

        out = F.pad(out, (3, 3), mode='constant', value=0)
        out = self.layer3(out)
        
        # Flatten output of convolutional layers
        out = out.view(out.size()[0], -1)
        
        # run fully connected layers
        out = self.layer4(out)
        out = self.layer5(out)
        predictions = self.layer6(out)
        
        activations, act_index = torch.max(activations, dim=2)
        
        return predictions, activations, act_index



def train_model(train_loader, test_loader, model, device, criterion, optimizer, num_epochs, output_directory):
    total_step = len(train_loader)
    model.train()

    #open files to log error
    train_error = open(output_directory + "training_error.txt", "a")
    test_error = open(output_directory + "test_error.txt", "a")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss_valid = float('inf')
    best_epoch = 1

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (seqs, labels) in enumerate(train_loader):
            seqs = seqs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs, act, idx = model(seqs)
            loss = criterion(outputs, labels) # change input to 
            running_loss += loss.item()
        
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        #save training loss to file
        epoch_loss = running_loss / len(train_loader.dataset)
        print("%s, %s" % (epoch, epoch_loss), file=train_error)

        #calculate test loss for epoch
        test_loss = 0.0
        with torch.no_grad():
            model.eval()
            for i, (seqs, labels) in enumerate(test_loader):
                x = seqs.to(device)
                y = labels.to(device)
                outputs, act, idx = model(x)
                loss = criterion(outputs, y)
                test_loss += loss.item() 

        test_loss = test_loss / len(test_loader.dataset)

        #save outputs for epoch
        print("%s, %s" % (epoch, test_loss), file=test_error)

        if test_loss < best_loss_valid:
            best_loss_valid = test_loss
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            print ('Saving the best model weights at Epoch [{}], Best Valid Loss: {:.4f}' 
                       .format(epoch+1, best_loss_valid))


    train_error.close()
    test_error.close()

    model.load_state_dict(best_model_wts)
    return model, best_loss_valid
    

def test_model(test_loader, model, device):
    num_filters=model.layer1_conv[0].out_channels
    predictions = torch.zeros(0, 81)
    max_activations = torch.zeros(0, num_filters)
    act_index = torch.zeros(0, num_filters)

    with torch.no_grad():
        model.eval()
        for seqs, labels in test_loader:
            seqs = seqs.to(device)
            pred, act, idx = model(seqs)
            predictions = torch.cat((predictions, pred.type(torch.FloatTensor)), 0)
            max_activations = torch.cat((max_activations, act.type(torch.FloatTensor)), 0)
            act_index = torch.cat((act_index, idx.type(torch.FloatTensor)), 0)

    predictions = predictions.numpy()
    max_activations = max_activations.numpy()
    act_index = act_index.numpy()
    return predictions, max_activations, act_index



def get_motifs(data_loader, model, device):
    num_filters=model.layer1_conv[0].out_channels
    activations = torch.zeros(0, num_filters, 251)
    predictions = torch.zeros(0, num_filters, 81)
    with torch.no_grad():
        model.eval()
        for seqs, labels in data_loader:
            seqs = seqs.to(device)
            pred, act, idx = model(seqs, num_filters)
            
            activations = torch.cat((activations, act.type(torch.FloatTensor)), 0)
            predictions = torch.cat((predictions, pred.type(torch.FloatTensor)), 0)
            
    predictions = predictions.numpy()
    activations = activations.numpy()
    return activations, predictions
