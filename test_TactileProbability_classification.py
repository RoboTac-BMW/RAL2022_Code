# from data_utils.ModelNetDataLoader import ModelNetDataLoader
# from data_utils.OFFDataLoader import *
import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
import sys
import importlib
import json
import csv
from path import Path
from data_utils.PCDLoader import *
from scipy.stats import entropy

from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size in training')
    # parser.add_argument('--num_category', default=10, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_category', default=12, type=int, help='training on real dataset')
    parser.add_argument('--num_point', type=int, default=50, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--SO3_Rotation', action='store_true', default=False, help='arbitrary rotation in SO3')
    parser.add_argument('--pcd_dir', type=str, default=None, help='The path of the tactile pcd')
    parser.add_argument('--tmp_label', type=str, default=None, help='Use only for saving file')
    return parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

# def test(model, loader, num_class=15, vote_num=1):
#     mean_correct = []
#     classifier = model.eval()
#     class_acc = np.zeros((num_class, 3))

#     for j, data in tqdm(enumerate(loader), total=len(loader)):
#         if not args.use_cpu:
#             # points, target = points.cuda(), target.cuda()
#             points, target = data['pointcloud'].to(device).float(), data['category'].to(device)
#             # print("points............")
#             # print(points.size())

#         points = points.transpose(2, 1)
#         vote_pool = torch.zeros(target.size()[0], num_class).cuda()

#         for _ in range(vote_num):
#             pred, _ = classifier(points)
#             vote_pool += pred
#         pred = vote_pool / vote_num
#         pred_choice = pred.data.max(1)[1]

#         for cat in np.unique(target.cpu()):
#             classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
#             class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
#             class_acc[cat, 1] += 1
#         correct = pred_choice.eq(target.long().data).cpu().sum()
#         mean_correct.append(correct.item() / float(points.size()[0]))

#     class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
#     class_acc = np.mean(class_acc[:, 2])
#     instance_acc = np.mean(mean_correct)
#     return instance_acc, class_acc

# def get_monte_carlo_predictions_Dataset(model,
#                                         data_loader,
#                                         forward_passes=3,
#                                         n_classes=15,
#                                         n_samples=6000):
#     """ Function to get the monte-carlo samples and uncertainty estimates
#     through multiple forward passes

#     Parameters
#     ----------
#     data_loader : object
#         data loader object from the data loader module
#     forward_passes : int
#         number of monte-carlo samples/forward passes
#     model : object
#         keras model
#     n_classes : int
#         number of classes in the dataset
#     n_samples : int
#         number of samples in the test set
#     """

#     dropout_predictions = np.empty((0, n_samples, n_classes))
#     softmax = nn.Softmax(dim=1)
#     for i in tqdm(range(forward_passes)):
#         # print(i)
#         predictions = np.empty((0, n_classes))
#         model.eval()
#         classifier = model.eval()
#         enable_dropout(model)
#         for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
#             if not args.use_cpu:
#                 points, target = data['pointcloud'].to(device).float(), data['category'].to(device)

#             points = points.transpose(2,1)
#             # image = image.to(torch.device('cuda'))

#             with torch.no_grad():
#                 # output = model(image)
#                 output, _ = classifier(points)
#                 output = softmax(output) # shape (n_samples, n_classes)
#                 # print(output)
#             predictions = np.vstack((predictions, output.cpu().numpy()))

#         dropout_predictions = np.vstack((dropout_predictions,
#                                          predictions[np.newaxis, :, :]))
#         # print(dropout_predictions)
#     mean = np.mean(dropout_predictions, axis=0)
#     print(mean.shape)
#     print(mean[100])
#         # dropout predictions - shape (forward_passes, n_samples, n_classes)


def get_monte_carlo_predictions(model,
                                data_loader,
                                forward_passes,
                                n_samples,
                                n_classes=12):
    """ Function to get the monte-carlo samples and uncertainty estimates
    through multiple forward passes

    Parameters
    ----------
    data_loader : object
        data loader object from the data loader module
    forward_passes : int
        number of monte-carlo samples/forward passes
    model : object
        keras model
    n_classes : int
        number of classes in the dataset
    n_samples : int
        number of samples in the test set
    """

    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)
    for i in range(forward_passes):
        # print(i)
        predictions = np.empty((0, n_classes))
        model.eval()
        classifier = model.eval()
        enable_dropout(model)
        for i, data in enumerate(data_loader):

            points = data.to(device).float()
            points = points.transpose(2,1)

            with torch.no_grad():
                output, _ = classifier(points)
                output = softmax(output) # shape (n_samples, n_classes)
                # print(output)
            predictions = np.vstack((predictions, output.cpu().numpy()))

        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))

        # print(dropout_predictions)
    mean = np.mean(dropout_predictions, axis=0)
    # print(mean.shape)
    # print(mean[0])
    prob_sample = mean[0]
    # entr_sample = entropy(prob_sample)
    return prob_sample




    # print(type(mean[0]))
    # mean = np.mean(dropout_predictions, axis=0)
    # print(mean.shape)
    # print(mean[100])
        # dropout predictions - shape (forward_passes, n_samples, n_classes)




def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # log_string('PARAMETER ...')
    # log_string(args)

    '''DATA LOADING'''
    # log_string('Load dataset ...')
    # visual_data_path = '/home/airocs/Desktop/active_vision_pcd_entropy/'
    # data_path = 'data/modelnet40_normal_resampled/'
    # data_path = Path("mesh_data/ModelNet10")
    offline_tactile_path = 'data/tactile_pcd_10_sampled_21.02'
    # tactile_data_dir = 'data/active_tactile_data'
    tactile_data_dir = args.pcd_dir

    # classes = find_classes(Path(visual_data_path))
    classes = ['cleaner', 'coffee', 'cup', 'eraser', 'glasses_box', 'jam', 'olive_oil', 'shampoo', 'spray', 'sugar', 'tape', 'wine']
    print(classes)
    # visual_pcd_files = []
    output_files = []

    # for category in classes.keys():
    #     new_dir = visual_data_path/Path(category)/'Train'
    for file in os.listdir(tactile_data_dir):
        if file.endswith('.pcd'):
            sample = {}
            sample['pcd_path'] = str((tactile_data_dir + str(file)))
            # sample['category'] = category
            sample['probability'] = 0.0
            output_files.append(sample)


    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals, mc_dropout=True)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # with torch.no_grad():
    #     instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
    #     log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

    # Load Samples
    print("...............................")
    print(len(output_files))


    for index, sample in tqdm(enumerate(output_files), total=len(output_files)):
        pcd_path = sample['pcd_path']
        pcd_dataset = PCDTest(pcd_path, sub_sample=True, sample_num=args.num_point)
        pcdDataLoader = torch.utils.data.DataLoader(pcd_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=False, num_workers=10)

        probability_sample = get_monte_carlo_predictions(classifier, pcdDataLoader,
                                             forward_passes=50, n_samples=1, n_classes=12)
        print(probability_sample)
        sample['probability'] = probability_sample

    sorted_sample_list = sorted(output_files, key=lambda x: x['pcd_path'], reverse=False)

    saved_file_path = "/home/airocs/Desktop/tactile_output_" + str(args.tmp_label) + str(datetime.now())  +".csv"
    with open(saved_file_path, 'w') as f:
        writer = csv.writer(f)
        # json.dump(classes, f)
        # f.write('\n')
        for item in sorted_sample_list:
            print(item)
            writer.writerow(item['probability'])
            # json.dump(item, f)
            # f.write('\n')

    print("File saved to %s " % saved_file_path)



if __name__ == '__main__':
    args = parse_args()
    main(args)
