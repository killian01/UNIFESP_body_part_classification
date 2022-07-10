import os
import numpy as np
import torchvision

import torch
import torch.nn as nn
import shutil
import cv2


class bodyPartClassifier(object):
    def __init__(self, model):
        self._model = model
        self._optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9,0.999), eps=0.0000001)
        
    def load_ckp(self, checkpoint_fpath):
        """
        checkpoint_path: path to save checkpoint      
        """
        # load check point
        checkpoint = torch.load(checkpoint_fpath)
        # initialize state_dict from checkpoint to model
        self._model.load_state_dict(checkpoint['state_dict'])
        # initialize optimizer from checkpoint to optimizer
        self._optimizer.load_state_dict(checkpoint['optimizer'])
        # initialize valid_loss_min from checkpoint to valid_loss_min
        valid_loss_min = checkpoint['valid_loss_min']

    

    def save_ckp(self, state, is_best, checkpoint_path, best_model_path):
        """
        state: checkpoint we want to save
        is_best: is this the best checkpoint; min validation loss
        checkpoint_path: path to save checkpoint
        best_model_path: path to save best model
        """
        f_path = checkpoint_path
        # save checkpoint data to the path given, checkpoint_path
        torch.save(state, f_path)
        # if it is a best model, min validation loss
        if is_best:
            best_fpath = best_model_path
            # copy that checkpoint file to best path given, best_model_path
            shutil.copyfile(f_path, best_fpath)
        

    

    def get_heatmap(self, pred, best_pred, img):
        # get the gradient of the output with respect to the parameters of the model
        pred[:, best_pred].backward()

        # pull the gradients out of the model
        gradients = self._model.get_activations_gradient()


        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = self._model.get_activations(img).detach()


        # weight the channels by corresponding gradients
        for i in range(512):
            activations[:, i, :, :] *= pooled_gradients[i]

        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap.cpu(), 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        #plt.matshow(heatmap.squeeze())
        return heatmap
    
    def get_sup_img(self, heatmap, images):
        #img_ar = img.numpy()
        #img_ar = img_ar[0]
        #img_ar = img_ar.transpose(1, 2, 0)

        img_ar = images[0].cpu().permute(1, 2, 0).numpy()
        img_ar = np.uint8(255 * img_ar)
        #img_ar = cv2.cvtColor(img_ar, cv2.COLOR_BGR2GRAY)

        #print(img_ar.shape[1])
        #plt.imshow(img_ar)
        #plt.show()
        #print(heatmap.shape)
        heatmap_ar = heatmap.numpy()
        #heatmap_ar = cv2.cvtColor(heatmap_ar, cv2.COLOR_RGB2BGR)


        heatmap_ar = cv2.resize(heatmap_ar, (img_ar.shape[1], img_ar.shape[0]))
        heatmap_ar = np.uint8(255 * heatmap_ar)
        heatmap_ar = cv2.applyColorMap(heatmap_ar, cv2.COLORMAP_JET)

        #superimposed_img = heatmap_ar *0.5 + img_ar
        superimposed_img = cv2.addWeighted(img_ar,0.8,heatmap_ar,0.5,0)

        img_float32 = np.float32(superimposed_img)
        im_rgb = cv2.cvtColor(img_float32, cv2.COLOR_RGB2BGR)
        #cv2.imwrite('./map.jpg', superimposed_img)

        #plt.imshow(im_rgb.astype('uint32'))
        #plt.show()

        return im_rgb


    def train_epoch(self, train_dataloader, loss_fn, device):
        losses = []
        correct_predictions = 0
        for data, labels, path in train_dataloader:
                data = data.to(device)
                labels = labels.to(device)
                self._optimizer.zero_grad()
                output = self._model(data.float())
                loss = loss_fn(output, labels)
                loss.backward()
                self._optimizer.step()
                losses.append(loss.item())
                predicted_labels = output.argmax(dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()
                del data
                del labels
                del path
                del output
                del loss


        accuracy = 100.0 * correct_predictions / len(train_dataloader.dataset)
        mean_loss = np.array(losses).mean()
        return accuracy, mean_loss
    
    def evaluate(self, dataloader, loss_fn, device):
        losses = []
        correct_predictions = 0
        predictions = []
        predicted_label = []
        labels_list = []

        with torch.no_grad():
            for images, labels, path in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                output = self._model(images)
                loss = loss_fn(output, labels)
                predicted_labels = output.argmax(dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()
                losses.append(float(loss.item()))
                del images
                del labels
                del path
                del output
                del loss


        mean_loss = np.array(losses).mean()
        accuracy = 100.0 * correct_predictions / len(dataloader.dataset)
        
        return accuracy, mean_loss

    def train(self, stop, start_epochs, valid_loss_min_input, checkpoint_path, bestmodel_path, train_dataloader, validation_dataloader, n_epochs, loss_fn=nn.CrossEntropyLoss(), device=torch.device('cuda')):
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        valid_loss_min = valid_loss_min_input

        count = 0
        stop_criterion = stop
        for epoch in range(start_epochs, n_epochs):
            self._model.train()
            train_accuracy, train_loss = self.train_epoch(train_dataloader, loss_fn, device)
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)

            self._model.eval()
            val_accuracy, val_loss = self.evaluate(validation_dataloader, loss_fn, device)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

            print('Epoch {}, train_loss: {}, train_accuracy: {}, validation_loss: {}, validation_accuracy: {}'.format(epoch+1, train_loss, train_accuracy, val_loss, val_accuracy))

            # create checkpoint variable and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': val_loss,
                'state_dict': self._model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
                #test
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                }
            # save checkpoint
            self.save_ckp(checkpoint, False, checkpoint_path, bestmodel_path)

            # save the model if validation loss has decreased
            if val_loss < valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,val_loss))
                # save checkpoint as best model
                self.save_ckp(checkpoint, True, checkpoint_path, bestmodel_path)
                valid_loss_min = val_loss
                count = 0
            else:
                count += 1
                if count == stop_criterion:
                    print('No improvement for {} epochs; training stopped.'.format(stop_criterion))
                    break
            if start_epochs == n_epochs:
                epoch = start_epochs

        return train_losses, val_losses, train_accuracies, val_accuracies, epoch+1
    
    def predict(self, dataloader, device):
        output_image = []
        predicted_label_list = []
        label_list = []
        path_list = []


        bad_class_images = []
        bad_class_labels = []
        bad_class_pred = []


        predictions = []
        predicted_label = []
        labels_list = []

        correct_predictions = 0

        self._model.eval()
        count = 0
        with torch.no_grad():
            for images, labels, path in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                output = self._model(images)
                predicted_labels = output.argmax(dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()

                if predicted_labels != labels:
                    bad_class_images.append(images.data.cpu().numpy())
                    bad_class_labels.append(labels.data.cpu().item())
                    bad_class_pred.append(predicted_labels.data.cpu().item())


                out = nn.functional.softmax(output, dim=1)
                predictions.append(out.data.cpu().numpy())
                predicted_label.append(predicted_labels.cpu().item())
                labels_list.append(labels.data.cpu().item())

                count += 1
            accuracy = 100.0 * correct_predictions / len(dataloader.dataset)

        return bad_class_pred, bad_class_labels, bad_class_images, predictions, predicted_label, labels_list, accuracy
    
    
    
    #Multi-labels functions
    def train_epoch_multi(self, train_dataloader, loss_fn, device):
        losses = []
        correct_predictions = 0
        for data, labels, path in train_dataloader:
                data = data.to(device)
                labels = labels.to(device)
                labels = labels.to(torch.float32)
                self._optimizer.zero_grad()
                output = self._model(data.float())
                loss = loss_fn(output, labels)
                loss.backward()
                self._optimizer.step()
                losses.append(loss.item())
                predicted_labels = output.round() 
                corr_pred_list = [1 if (predicted_labels[i, :] == labels[i,:]).all() else 0 for i in range(labels.shape[0])]
                correct_predictions += np.sum(corr_pred_list)

        accuracy = 100.0 * correct_predictions / len(train_dataloader.dataset)
        mean_loss = np.array(losses).mean()
        return accuracy, mean_loss

    def evaluate_multi(self, dataloader, loss_fn, device):
        losses = []
        correct_predictions = 0
        predictions = []
        predicted_label = []
        labels_list = []

        with torch.no_grad():
            for images, labels, path in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.to(torch.float32)

                output = self._model(images.float())
                loss = loss_fn(output, labels)
                predicted_labels = output.round() 
                corr_pred_list = [1 if (predicted_labels[i, :] == labels[i,:]).all() else 0 for i in range(labels.shape[0])]
                correct_predictions += np.sum(corr_pred_list)
                losses.append(loss.item())


        mean_loss = np.array(losses).mean()
        accuracy = 100.0 * correct_predictions / len(dataloader.dataset)
        return accuracy, mean_loss

    def train_multi(self, start_epochs, valid_loss_min_input, checkpoint_path, bestmodel_path, train_dataloader, validation_dataloader, n_epochs, loss_fn=nn.BCELoss(), device=torch.device('cuda')):

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        valid_loss_min = valid_loss_min_input

        count = 0
        stop_criterion = 5
        for epoch in range(start_epochs, n_epochs):
            self._model.train()
            train_accuracy, train_loss = self.train_epoch_multi(train_dataloader, loss_fn, device)
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)

            self._model.eval()
            val_accuracy, val_loss = self.evaluate_multi(validation_dataloader, loss_fn, device)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

            print('Epoch {}, train_loss: {}, train_accuracy: {}, validation_loss: {}, validation_accuracy: {}'.format(epoch+1, train_loss, train_accuracy, val_loss, val_accuracy))

            # create checkpoint variable and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': val_loss,
                'state_dict': self._model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
                #test
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                }
            # save checkpoint
            self.save_ckp(checkpoint, False, checkpoint_path, bestmodel_path)

            # save the model if validation loss has decreased
            if np.round(val_loss,3) < np.round(valid_loss_min, 3):
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,val_loss))
                # save checkpoint as best model
                self.save_ckp(checkpoint, True, checkpoint_path, bestmodel_path)
                valid_loss_min = val_loss
                count = 0
            else:
                count += 1
                if count == stop_criterion:
                    print('No improvement for {} epochs; training stopped.'.format(stop_criterion))
                    break
            if start_epochs == n_epochs:
                epoch = start_epochs    

        return train_losses, val_losses, train_accuracies, val_accuracies, epoch+1
    

    
    def predict_multi(self, dataloader, device):
        output_image = []
        predicted_label_list = []
        label_list = []
        path_list = []

        predictions = []
        predicted_label = []
        labels_list = []

        correct_predictions = 0

        self._model.eval()
        count = 0
        with torch.no_grad():
            for images, labels, path in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                output = self._model(images)
                
                predicted_labels = output.round() 
                corr_pred_list = [1 if (predicted_labels[i, :] == labels[i,:]).all() else 0 for i in range(labels.shape[0])]
                correct_predictions += np.sum(corr_pred_list)


                predictions.append(output.data.cpu().numpy())
                predicted_label.append(predicted_labels.cpu().numpy())
                labels_list.append(labels.data.cpu().numpy())

                count += 1
            accuracy = 100.0 * correct_predictions / len(dataloader.dataset)

        return predictions, predicted_label, labels_list, accuracy
    
    

    #MIL functions
    def train_epoch_MIL(self, trained_model, dataloader, loss_fn, device):
        losses = []
        correct_predictions = 0
        for images, labels, patient in (dataloader): 
            img_max = 17
            feature_vector = np.zeros((1, img_max*2048))

            with torch.no_grad():
                trained_model._model.eval()
                for i in range(len(images)):      
                    img = images[i].to(device)
                    output = trained_model._model.get_feature_vector(img)
                    feature_vector[0, i*2048:(i+1)*2048] = output.cpu().numpy()
                    del output
            
            feature_vector = torch.tensor(feature_vector)
            data = feature_vector.to(device)
            labels = labels.to(device)
            self._optimizer.zero_grad()
            output = self._model(data.float())
            loss = loss_fn(output, labels)
            loss.backward()
            self._optimizer.step()
            losses.append(loss.item())
            predicted_labels = output.argmax(dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            del output
            del feature_vector
            del data
            del labels
            del loss
            
        accuracy = 100.0 * correct_predictions / len(dataloader.dataset)
        mean_loss = np.array(losses).mean()
        
        return accuracy, mean_loss

    def evaluate_MIL(self, trained_model, dataloader, loss_fn, device):
        losses = []
        correct_predictions = 0
        predictions = []
        predicted_label = []
        labels_list = []

        for images, labels, patient in (dataloader): 
            img_max = 17
            feature_vector = np.zeros((1, img_max*2048))
            with torch.no_grad():
                trained_model._model.eval()
                for i in range(len(images)):      
                    img = images[i].to(device)
                    output = trained_model._model.get_feature_vector(img)
                    feature_vector[0, i*2048:(i+1)*2048] = output.cpu().numpy()
                    del output

            with torch.no_grad():
                feature_vector = torch.tensor(feature_vector)
                data = feature_vector.to(device)
                labels = labels.to(device)
                output = self._model(data.float())
                loss = loss_fn(output, labels)
                losses.append(loss.item())
                predicted_labels = output.argmax(dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()
                del output
                del feature_vector
                del data
                del labels
                del loss

        mean_loss = np.array(losses).mean()
        accuracy = 100.0 * correct_predictions / len(dataloader.dataset)
        return accuracy, mean_loss

    def train_MIL(self, trained_model, start_epochs, valid_loss_min_input, checkpoint_path, bestmodel_path, train_dataloader, validation_dataloader, n_epochs, loss_fn=nn.CrossEntropyLoss(), device=torch.device('cuda')):

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        valid_loss_min = valid_loss_min_input

        count = 0
        stop_criterion = 5
        for epoch in range(start_epochs, n_epochs):
            self._model.train()
            train_accuracy, train_loss = self.train_epoch_MIL(trained_model, train_dataloader, loss_fn, device)
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)

            self._model.eval()
            val_accuracy, val_loss = self.evaluate_MIL(trained_model, validation_dataloader, loss_fn, device)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

            print('Epoch {}, train_loss: {}, train_accuracy: {}, validation_loss: {}, validation_accuracy: {}'.format(epoch+1, train_loss, train_accuracy, val_loss, val_accuracy))

            # create checkpoint variable and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': val_loss,
                'state_dict': self._model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
                #test
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                }
            # save checkpoint
            self.save_ckp(checkpoint, False, checkpoint_path, bestmodel_path)

            # save the model if validation loss has decreased
            if np.round(val_loss,3) < np.round(valid_loss_min, 3):
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,val_loss))
                # save checkpoint as best model
                self.save_ckp(checkpoint, True, checkpoint_path, bestmodel_path)
                valid_loss_min = val_loss
                count = 0
            else:
                count += 1
                if count == stop_criterion:
                    print('No improvement for {} epochs; training stopped.'.format(stop_criterion))
                    break
            if start_epochs == n_epochs:
                epoch = start_epochs    

        return train_losses, val_losses, train_accuracies, val_accuracies, epoch+1
    

    
    def predict_MIL(self, trained_model, dataloader, device):
        output_image = []
        predicted_label_list = []
        label_list = []
        path_list = []


        bad_class_images = []
        bad_class_labels = []
        bad_class_pred = []


        predictions = []
        predicted_label = []
        labels_list = []

        correct_predictions = 0

        self._model.eval()
        
        for images, labels, patient in (dataloader): 
            img_max = 17
            feature_vector = np.zeros((1, img_max*2048))
            with torch.no_grad():
                trained_model._model.eval()
                for i in range(len(images)):      
                    img = images[i].to(device)
                    output = trained_model._model.get_feature_vector(img)
                    feature_vector[0, i*2048:(i+1)*2048] = output.cpu().numpy()
                    del output

            with torch.no_grad():
                feature_vector = torch.tensor(feature_vector)
                data = feature_vector.to(device)
                labels = labels.to(device)
                output = self._model(data.float())
                predicted_labels = output.argmax(dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()
                
                if predicted_labels != labels:
                    bad_class_images.append(data.data.cpu().numpy())
                    bad_class_labels.append(labels.data.cpu().item())
                    bad_class_pred.append(predicted_labels.data.cpu().item())


                out = nn.functional.softmax(output, dim=1)
                predictions.append(out.data.cpu().numpy())
                predicted_label.append(predicted_labels.cpu().item())
                labels_list.append(labels.data.cpu().item())

                del output
                del feature_vector
                del data
                del labels

        accuracy = 100.0 * correct_predictions / len(dataloader.dataset)

        return bad_class_pred, bad_class_labels, bad_class_images, predictions, predicted_label, labels_list, accuracy
            

        
        
    #Functions to train and evaluate a linear model with 2 features vector
    def train_epoch_2_instances(self, break_model_trained, RD_model_trained, dataloader, loss_fn, device):
        losses = []
        correct_predictions = 0
        for images, labels, patient in (dataloader): 
            feature_vector = np.zeros((1, 4096))
            img = images.to(device)
            
#             with torch.no_grad():
#                 trained_model._model.eval()
#                 for i in range(len(images)):      
#                     img = images[i].to(device)
#                     output = trained_model._model.get_feature_vector(img)
#                     feature_vector[0, i*2048:(i+1)*2048] = output.cpu().numpy()
#                     del output

            with torch.no_grad():
                break_model_trained._model.eval()
                RD_model_trained._model.eval()
                break_features = break_model_trained._model.get_feature_vector(img)
                RD_features = RD_model_trained._model.get_feature_vector(img)
                feature_vector[0, 0:2048] = break_features.cpu().numpy()
                feature_vector[0, 2048:4096] = RD_features.cpu().numpy()
                del break_features
                del RD_features
            
            feature_vector = torch.tensor(feature_vector)
            data = feature_vector.to(device)
            labels = labels.to(device)
            self._optimizer.zero_grad()
            output = self._model(data.float())
            loss = loss_fn(output, labels)
            loss.backward()
            self._optimizer.step()
            losses.append(loss.item())
            predicted_labels = output.argmax(dim=1)
            correct_predictions += (predicted_labels == labels).sum().item()
            del output
            del feature_vector
            del data
            del labels
            del loss
            
        accuracy = 100.0 * correct_predictions / len(dataloader.dataset)
        mean_loss = np.array(losses).mean()
        
        return accuracy, mean_loss

    def evaluate_2_instances(self, break_model_trained, RD_model_trained, dataloader, loss_fn, device):
        losses = []
        correct_predictions = 0
        predictions = []
        predicted_label = []
        labels_list = []

        break_model_trained._model.eval()
        RD_model_trained._model.eval()
        for images, labels, patient in (dataloader): 
            feature_vector = np.zeros((1, 4096))
            img = images.to(device)
#             with torch.no_grad():
#                 trained_model._model.eval()
#                 for i in range(len(images)):      
#                     img = images[i].to(device)
#                     output = trained_model._model.get_feature_vector(img)
#                     feature_vector[0, i*2048:(i+1)*2048] = output.cpu().numpy()
#                     del output
            with torch.no_grad():
                break_features = break_model_trained._model.get_feature_vector(img)
                RD_features = RD_model_trained._model.get_feature_vector(img)

                feature_vector[0, 0:2048] = break_features.cpu().numpy()
                feature_vector[0, 2048:4096] = RD_features.cpu().numpy()
                del break_features
                del RD_features


            with torch.no_grad():
                feature_vector = torch.tensor(feature_vector)
                data = feature_vector.to(device)
                labels = labels.to(device)
                output = self._model(data.float())
                loss = loss_fn(output, labels)
                losses.append(loss.item())
                predicted_labels = output.argmax(dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()
                del output
                del feature_vector
                del data
                del labels
                del loss

        mean_loss = np.array(losses).mean()
        accuracy = 100.0 * correct_predictions / len(dataloader.dataset)
        return accuracy, mean_loss

    def train_2_instances(self, break_model_trained, RD_model_trained, start_epochs, valid_loss_min_input, checkpoint_path, bestmodel_path, train_dataloader, validation_dataloader, n_epochs, loss_fn=nn.CrossEntropyLoss(), device=torch.device('cuda')):

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        valid_loss_min = valid_loss_min_input

        count = 0
        stop_criterion = 5
        for epoch in range(start_epochs, n_epochs):
            self._model.train()
            train_accuracy, train_loss = self.train_epoch_2_instances(break_model_trained, RD_model_trained, train_dataloader, loss_fn, device)
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)

            self._model.eval()
            val_accuracy, val_loss = self.evaluate_2_instances(break_model_trained, RD_model_trained, validation_dataloader, loss_fn, device)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

            print('Epoch {}, train_loss: {}, train_accuracy: {}, validation_loss: {}, validation_accuracy: {}'.format(epoch+1, train_loss, train_accuracy, val_loss, val_accuracy))

            # create checkpoint variable and add important data
            checkpoint = {
                'epoch': epoch + 1,
                'valid_loss_min': val_loss,
                'state_dict': self._model.state_dict(),
                'optimizer': self._optimizer.state_dict(),
                #test
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                }
            # save checkpoint
            self.save_ckp(checkpoint, False, checkpoint_path, bestmodel_path)

            # save the model if validation loss has decreased
            if np.round(val_loss,3) < np.round(valid_loss_min, 3):
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,val_loss))
                # save checkpoint as best model
                self.save_ckp(checkpoint, True, checkpoint_path, bestmodel_path)
                valid_loss_min = val_loss
                count = 0
            else:
                count += 1
                if count == stop_criterion:
                    print('No improvement for {} epochs; training stopped.'.format(stop_criterion))
                    break
            if start_epochs == n_epochs:
                epoch = start_epochs    

        return train_losses, val_losses, train_accuracies, val_accuracies, epoch+1
    

    
    def predict_2_instances(self, break_model_trained, RD_model_trained, dataloader, device):
        output_image = []
        predicted_label_list = []
        label_list = []
        path_list = []


        bad_class_images = []
        bad_class_labels = []
        bad_class_pred = []


        predictions = []
        predicted_label = []
        labels_list = []

        correct_predictions = 0


        self._model.eval()
        for images, labels, patient in (dataloader): 
            feature_vector = np.zeros((1, 4096))
            img = images.to(device)
            with torch.no_grad():
                break_model_trained._model.eval()
                RD_model_trained._model.eval()
#                 for i in range(len(images)):      
#                     img = images[i].to(device)
#                     output = trained_model._model.get_feature_vector(img)
#                     feature_vector[0, i*2048:(i+1)*2048] = output.cpu().numpy()
#                     del output
                break_features = break_model_trained._model.get_feature_vector(img)
                RD_features = RD_model_trained._model.get_feature_vector(img)

                feature_vector[0, 0:2048] = break_features.cpu().numpy()
                feature_vector[0, 2048:4096] = RD_features.cpu().numpy()
                del break_features
                del RD_features
                
            with torch.no_grad():
                feature_vector = torch.tensor(feature_vector)
                data = feature_vector.to(device)
                labels = labels.to(device)
                output = self._model(data.float())
                predicted_labels = output.argmax(dim=1)
                correct_predictions += (predicted_labels == labels).sum().item()
                
                if predicted_labels != labels:
                    bad_class_images.append(data.data.cpu().numpy())
                    bad_class_labels.append(labels.data.cpu().item())
                    bad_class_pred.append(predicted_labels.data.cpu().item())


                out = nn.functional.softmax(output, dim=1)
                predictions.append(out.data.cpu().numpy())
                predicted_label.append(predicted_labels.cpu().item())
                labels_list.append(labels.data.cpu().item())

                del output
                del feature_vector
                del data
                del labels

        accuracy = 100.0 * correct_predictions / len(dataloader.dataset)

        return bad_class_pred, bad_class_labels, bad_class_images, predictions, predicted_label, labels_list, accuracy