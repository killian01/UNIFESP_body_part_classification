import numpy as np
import h5py
import os
import sys
import torchvision
from torchvision.transforms import *
from torch.utils.data import DataLoader
from sklearn import preprocessing
import warnings
import shutup


sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))

from body_part_dataset import bodyPartDataset
from CNN_models import Resnet50Multilabels
from body_part_classifier import bodyPartClassifier


def main():
    shutup.please()
    # warnings.filterwarnings("ignore", category=FutureWarning)
    main_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    data_path = main_dir + '\Data'

    Folder = h5py.File(data_path + "/data.hdf5", "r+")
    directories = np.char.decode(Folder["/Directories"][()])
    labels = np.char.decode(Folder["/Labels"][()])
    Folder.close()

    multi_labels = []
    for lab in labels:
        lab = lab[1:-1]
        label_list = lab.split(', ')
        label_list = [int(label) for label in label_list]
        multi_labels.append(label_list)

    lb = preprocessing.LabelBinarizer()
    lb.fit(np.arange(0, 22, 1))

    multi_labels = [np.sum(lb.transform(lab), axis=0) if len(lab) > 1 else lb.transform(lab)[0] for lab in multi_labels]


    target_size = (512, 512)
    transforms_train = Compose([
                        # ToPILImage(),
                        Resize(target_size),
                        RandomHorizontalFlip(),
                        RandomVerticalFlip(),
                        # ColorJitter(),
                        # RandomAffine(),
                        ToTensor(),
                        ])
    transforms_valid = Compose([
                        # ToPILImage(),
                        Resize(target_size),
                        ToTensor(),
                        ])

    train_data = directories[:int(0.6*len(directories))]
    valid_data = directories[int(0.6*len(directories)):int(0.8*len(directories))]
    test_data = directories[int(0.8*len(directories)):]

    train_labels = multi_labels[:int(0.6 * len(multi_labels))]
    valid_labels = multi_labels[int(0.6 * len(multi_labels)):int(0.8 * len(multi_labels))]
    test_labels = multi_labels[int(0.8 * len(multi_labels)):]

    train_data = bodyPartDataset(train_data, train_labels, transforms_train)
    valid_data = bodyPartDataset(valid_data, valid_labels, transforms_valid)
    test_data = bodyPartDataset(test_data, test_labels, transforms_valid)

    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)


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

    checkpoint_path = checkpoint_dir + '/check_model_multi_label_vector.pt'
    bestmodel_dir = os.path.join(main_dir, 'best_model/Multi_label_vector')
    if not os.path.exists(bestmodel_dir):
        os.mkdir(bestmodel_dir)

    bestmodel_path = bestmodel_dir + '/best_model_multi_label_vector.pt'
    result_path = os.path.join(main_dir, 'Results/Multi_label_vector')
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    n_epochs = 1
    train_losses, val_losses, train_acc, val_acc, n_epochs = classifier.train_multi_labels(start_epochs,
                                                                                           valid_loss_min,
                                                                                           checkpoint_path,
                                                                                           bestmodel_path,
                                                                                           train_dataloader,
                                                                                           valid_dataloader,
                                                                                           n_epochs,
                                                                                           loss_fn=loss_fn,
                                                                                           device=device)

    file = open(result_path + '/multi_label_vector', 'w')
    file.write('train_losses:' + str(train_losses) + '\n' +
               'val_losses:' + str(val_losses) + '\n' +
               'train_acc:' + str(train_acc) + '\n' +
               'val_acc:' + str(val_acc) + '\n' +
               'n_epochs:' + str(n_epochs))

    file.close()

    print('ok')


if __name__ == "__main__":
    main()
