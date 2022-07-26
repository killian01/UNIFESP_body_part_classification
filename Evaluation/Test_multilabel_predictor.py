import numpy as np
import h5py
import os
import sys
from torchvision.transforms import *
from torch.utils.data import DataLoader
from sklearn import preprocessing
import shutup
import pandas as pd

sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))

from body_part_dataset import bodyPartDataset
from CNN_models import Resnet50Multilabels
from body_part_classifier import bodyPartClassifier


def main(batch, model_name):
    main_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    data_path = main_dir + '\Data'

    Folder = h5py.File(data_path + "/test_data.hdf5", "r+")
    directories = np.char.decode(Folder["/Directories"][()])
    UIDs = np.char.decode(Folder["/UIDS"][()])
    Folder.close()

    lb = preprocessing.LabelBinarizer()
    lb.fit(np.arange(0, 22, 1))

    target_size = (512, 512)
    transforms_valid = Compose([
                        Resize(target_size),
                        ToTensor(),
                        ])
    labels = [0 for idx in range(len(directories))]
    test_data = bodyPartDataset(directories, labels, transforms_valid)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    device = torch.device('cuda')
    model = Resnet50Multilabels()
    model = model.to(device)
    classifier = bodyPartClassifier(model)

    bestmodel_dir = os.path.join(main_dir, 'best_model/Multi_label_vector')
    bestmodel_path = bestmodel_dir + '/' + model_name + '.pt'
    classifier.load_ckp(bestmodel_path)

    predictions, predicted_label, labels_list = classifier.predict_multi_labels(test_dataloader, device)

    predicted_label_flat = [pred[0] for pred in predicted_label]
    df_pred = pd.DataFrame(predicted_label_flat)
    label_list = []
    for idx, df_row in df_pred.iterrows():
        labels = df_row.loc[df_row == 1]
        label_list.append(labels.index)
    label_list = [str(list(lab))[1:-1] for lab in label_list]
    label_list = [lab.replace(",", "") for lab in label_list]

    UID = [uid[:-6] for uid in UIDs]
    df_results = pd.DataFrame((UID), columns=['SOPInstanceUID'])
    df_results['Target'] = label_list

    df_results.to_csv(main_dir + '/Evaluation/results.csv', index=False)

    print('ok')


if __name__ == "__main__":
    main(batch=8, model_name='Resnet50')
