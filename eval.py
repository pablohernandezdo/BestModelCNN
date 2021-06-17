import os
import time
import argparse

import numpy as np
import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader
from humanfriendly import format_timespan

from models import *
from dataset import NpyDataset


def main():
    # Measure exec time
    start_time = time.time()

    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",
                        default='STEAD-STEAD',
                        help="Name of dataset to evaluate on")
    parser.add_argument("--model_name", default='1h6k_test_model',
                        help="Name of model to eval")
    parser.add_argument("--model_folder", default='models',
                        help="Model to eval folder")
    parser.add_argument("--classifier", default='1h6k',
                        help="Choose classifier architecture")
    parser.add_argument("--test_path", default='Test_data_v2.hdf5',
                        help="HDF5 test Dataset path")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Size of the training batches")
    args = parser.parse_args()

    # Select training device
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Test dataset
    test_set = NpyDataset(args.test_path)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=8)

    # Load specified Classifier
    net = get_classifier(args.classifier)
    net.to(device)

    # Count number of parameters
    params = count_parameters(net)

    # Load from trained model
    net.load_state_dict(torch.load(f'{args.model_folder}/'
                                   f'{args.model_name}.pth'))
    net.eval()

    # Evaluate model on test set
    evaluate_dataset(test_loader, args.dataset_name + '/Test',
                     device, net, args.model_name,
                     args.model_folder,
                     'Net_outputs')

    eval_end = time.time()
    total_time = eval_end - start_time

    print(f'Total evaluation time: {format_timespan(total_time)}\n\n'
          f'Number of network parameters: {params}')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_dataset(data_loader, dataset_name, device, net,
                     model_name, model_folder, csv_folder):

    # List of outputs and labels used to create pd dataframe
    dataframe_rows_list = []

    with tqdm.tqdm(total=len(data_loader),
                   desc=f'{dataset_name} dataset evaluation') as data_bar:

        with torch.no_grad():
            for data in data_loader:

                traces, labels = data[0].to(device), data[1].to(device)
                outputs = net(traces)

                for out, lab in zip(outputs, labels):
                    new_row = {'out': out.item(),
                               'label': int(lab.item())}
                    dataframe_rows_list.append(new_row)

                data_bar.update(1)

    test_outputs = pd.DataFrame(dataframe_rows_list)

    if len(model_folder.split("/")) > 1:
        model_dirname = model_folder.split("/")[-1]

    else:
        model_dirname = ""

    # Create csv folder if necessary
    save_folder = f'{csv_folder}/{dataset_name}/{model_dirname}'

    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    # Save outputs and labels to csv file
    test_outputs.to_csv(f'{save_folder}/{model_name}.csv', index=False)


def get_classifier(x):
    print(x)
    if x == 'CNN1':
        return CNN1()
    elif x == 'CNN2':
        return CNN2()
    elif x == 'CNN3':
        return CNN3()


# def get_classifier(x):
#     if x == 'CNN_6k_6k':
#         return CNN_6k_6k()
#     elif x == 'CNN_6k_5k':
#         return CNN_6k_5k()
#     elif x == 'CNN_6k_4k':
#         return CNN_6k_4k()
#     elif x == 'CNN_6k_3k':
#         return CNN_6k_3k()
#     elif x == 'CNN_6k_2k':
#         return CNN_6k_2k()
#     elif x == 'CNN_6k_1k':
#         return CNN_6k_1k()
#     elif x == 'CNN_5k_5k':
#         return CNN_5k_5k()
#     elif x == 'CNN_5k_4k':
#         return CNN_5k_4k()
#     elif x == 'CNN_5k_3k':
#         return CNN_5k_3k()
#     elif x == 'CNN_5k_2k':
#         return CNN_5k_2k()
#     elif x == 'CNN_5k_1k':
#         return CNN_5k_1k()
#     elif x == 'CNN_4k_4k':
#         return CNN_4k_4k()
#     elif x == 'CNN_4k_3k':
#         return CNN_4k_3k()
#     elif x == 'CNN_4k_2k':
#         return CNN_4k_2k()
#     elif x == 'CNN_4k_1k':
#         return CNN_4k_1k()
#     elif x == 'CNN_3k_3k':
#         return CNN_3k_3k()
#     elif x == 'CNN_3k_2k':
#         return CNN_3k_2k()
#     elif x == 'CNN_3k_1k':
#         return CNN_3k_1k()
#     elif x == 'CNN_2k_2k':
#         return CNN_2k_2k()
#     elif x == 'CNN_2k_1k':
#         return CNN_2k_1k()
#     elif x == 'CNN_1k_1k':
#         return CNN_1k_1k()


if __name__ == "__main__":
    main()
