# from data_utils.ModelNetDataLoader import ModelNetDataLoader
# from data_utils.OFFDataLoader import *
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import sys
import importlib
from path import Path
from data_utils.PCDLoader import *

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    # parser.add_argument('--num_category', default=10, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_category', default=15, type=int, help='training on real dataset')
    parser.add_argument('--sample_point', type=bool, default=True,  help='Sampling on tacitle data')
    parser.add_argument('--num_point', type=int, default=80, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--SO3_Rotation', action='store_true', default=False, help='arbitrary rotation in SO3')
    return parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(model, loader, num_class=15, vote_num=1):
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))
    print(len(loader))
    y_pred = []
    y_true = []

    for j, data in tqdm(enumerate(loader), total=len(loader)):
        if not args.use_cpu:
            # points, target = points.cuda(), target.cuda()
            points, target = data['pointcloud'].to(device).float(), data['category'].to(device)
            # print(target)
            # print("points............")
            # print(points.size())

        points = points.transpose(2, 1)
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        # print(pred.data.max(1)[1])
        pred_choice = pred.data.max(1)[1]

        # pred for confusion matrix
        pred_conf = (torch.max(torch.exp(pred), 1)[1]).data.cpu().numpy()
        y_pred.extend(pred_conf)
        y_true.extend(target.data.cpu())

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            # print("------------------------------------------------------")
            # print("cat", cat)
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))


    # print(mean_correct)
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    # print(instance_acc)
    # Draw Confusion Matrix
    # print(y_true)
    # print(y_pred)
    cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')
    # print(cf_matrix)
    return instance_acc, class_acc, cf_matrix
    # return instance_acc, class_acc


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
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # tactile_data_path = 'data/tactile_data_pcd/'
    tactile_data_path = 'data/test_tactile_data_pcd/'
    # tactile_data_path = 'data/visual_data_pcd/'
    # data_path = 'data/modelnet40_normal_resampled/'
    # data_path = Path("mesh_data/ModelNet10")


    test_dataset = PCDPointCloudData(tactile_data_path,
                                     folder='Train',
                                     sample_method='Voxel',
                                     num_point=args.num_point,
                                     sample=args.sample_point,
                                     est_normal=args.use_normals,
                                     rotation=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    # Load labels:
    classes = find_classes(tactile_data_path)
    print(classes)
    print(classes.keys)


    with torch.no_grad():
        # instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        instance_acc, class_acc, cf_matrix = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))

        # Draw confusion matrix
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10,
                             index = [i for i in classes.keys()], columns = [i for i in classes.keys()])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(experiment_dir + '/' + str(datetime.now()) + '.png')

if __name__ == '__main__':
    args = parse_args()
    main(args)
