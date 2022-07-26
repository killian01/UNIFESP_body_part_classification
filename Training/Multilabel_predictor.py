import numpy as np
import h5py
import os
import sys
from torchvision.transforms import *
from torch.utils.data import DataLoader
from sklearn import preprocessing
import shutup


sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))

from body_part_dataset import bodyPartDataset
from CNN_models import Resnet50Multilabels
from body_part_classifier import bodyPartClassifier


def main(run_name, n_epochs, batch, stop_criterion, load_checkpoint):
    shutup.please()
    main_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    data_path = main_dir + '\Data'

    Folder = h5py.File(data_path + "/Folder_0.hdf5", "r+")
    X_train = np.char.decode(Folder["/X_train"][()])
    X_valid = np.char.decode(Folder["/X_valid"][()])
    y_train = np.char.decode(Folder["/y_train"][()])
    y_valid = np.char.decode(Folder["/y_valid"][()])
    Folder.close()

    target_size = (512, 512)
    transforms_train = Compose([
                        Resize(target_size),
                        RandomHorizontalFlip(),
                        RandomVerticalFlip(),
                        RandomAffine(45),
                        ToTensor(),
                        ])
    transforms_valid = Compose([
                        Resize(target_size),
                        ToTensor(),
                        ])

    train_data = bodyPartDataset(X_train, y_train, transforms_train)
    valid_data = bodyPartDataset(X_valid, y_valid, transforms_valid)

    train_dataloader = DataLoader(train_data, batch_size=batch, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)


    start_epochs = 0
    valid_loss_min = np.inf
    device = torch.device('cuda')
    model = Resnet50Multilabels()
    model = model.to(device)

    loss_fn = torch.nn.BCELoss()
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=0.0000001)
    classifier = bodyPartClassifier(model)

    checkpoint_dir = os.path.join(main_dir, 'checkpoints')
    checkpoint_multi_label_dir = os.path.join(main_dir, 'checkpoints/Multi_label_vector')
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    if not os.path.exists(checkpoint_multi_label_dir):
        os.mkdir(checkpoint_multi_label_dir)
    checkpoint_path = checkpoint_multi_label_dir + '/' + str(run_name) + '.pt'

    bestmodel_dir = os.path.join(main_dir, 'best_model/Multi_label_vector')
    if not os.path.exists(bestmodel_dir):
        os.mkdir(bestmodel_dir)
    bestmodel_path = bestmodel_dir + '/' + str(run_name) + '.pt'

    result_path = os.path.join(main_dir, 'Results/Multi_label_vector')
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    if load_checkpoint:
        classifier.load_ckp(checkpoint_path)

    train_losses, val_losses, train_acc, val_acc, n_epochs = classifier.train_multi_labels(start_epochs,
                                                                                           valid_loss_min,
                                                                                           checkpoint_path,
                                                                                           bestmodel_path,
                                                                                           train_dataloader,
                                                                                           valid_dataloader,
                                                                                           n_epochs,
                                                                                           stop_criterion,
                                                                                           loss_fn=loss_fn,
                                                                                           device=device)

    if load_checkpoint is False:
        file = open(result_path + '/' + str(run_name) + '.pt', 'w')
    else:
        file = open(result_path + '/' + str(run_name) + '_next.pt', 'w')

    file.write('train_losses:' + str(train_losses) + '\n' +
               'val_losses:' + str(val_losses) + '\n' +
               'train_acc:' + str(train_acc) + '\n' +
               'val_acc:' + str(val_acc) + '\n' +
               'n_epochs:' + str(n_epochs))
    file.close()


if __name__ == "__main__":
    main(run_name='Resnet50_fold_0',
         n_epochs=25,
         batch=8,
         stop_criterion=5,
         load_checkpoint=False)
