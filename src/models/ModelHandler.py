"""File to train and test model and extract motifs"""
import torch
import copy

class ModelHandler():
    """
    Class to train and test model and extract motifs
    """

    def train_model(self, train_loader, test_loader, model, device, criterion, optimizer, num_epochs, output_directory):
        """
        Train model
        """
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
        

    def test_model(self, test_loader, model, device):
        """
        Test model
        """
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



    def get_motifs(self, data_loader, model, device):
        """
        Extract motifs
        """
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
