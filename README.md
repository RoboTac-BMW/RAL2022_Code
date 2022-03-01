# Pytorch Implementation

Thanks for the original author https://github.com/yanx27/Pointnet_Pointnet2_pytorch

## Branch Instructions

**Important branches:**
| Branch Name                                                                          | Details                                                                                              | Parameters                                                   |
|--------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| cong_dev                                                                             | Main branch. Containing all the latest code.                                                         | ---                                                          |
| PointNet-40-Dense-WithoutNormal                                                      | pre-trained network with ModelNet40 without normals                                                  | learning rate 0.001, epoch 300                               |
| PointNet-40-Dense-Normal                                                             | pre-trained network with ModelNet40 with normals                                                     | learning rate 0.001, epoch 300                               |
| PointNet-Dense-WithoutNormal                                                         | pre-trained network with ModelNet10 without normals                                                  | learning rate 0.001, epoch 150                               |
| PointNet-Dense-Normal                                                                | pre-trained network with ModelNet10 with normals                                                     | learning rate 0.001, epoch 150                               |
| RealData-40-Dense-WithoutNormal                                                      | Fine-tuning the best model in PointNet-40-Dense-WithoutNormal with real vision dataset (15 classes)  | learning rate 0.001, epoch 100                               |
| RealData-12-40-Dense-WithoutNormal                                                   | Fine-tuning the best model in PointNet-40-Dense-WithoutNormal with real vision dataset (12 classes)  | learning rate 0.001, epoch 50                                |
| RealData-12-40-MultiLayer-Loss-alpha_10-beta_10-lamda_10-50-learning_rate_0.0001-new | [Best Model!] DA training with the best model in RealData-12-40-Dense-WithoutNormal (12 classes).    | alpha 10, beta 10, lamda 10, learning rate 0.0001, epoch 50  |
| RealData-Scratch-Dense-WithoutNormal                                                 | Trained from scratch with real vision dataset (15 classes)                                           | learning rate 0.001, epoch 100                               |
| RealData-Scratch-ActiveVision-2500-Selected                                          | Active Vision trained with the selected highest entropy samples 500-4500.                            | learning rate 0.001, epoch 20 each 500 samples               |
| RealData-Scratch-ActiveVision-2000-Reversed                                          | Active Vision trained with the lowest entropy samples 500-4500.                                      | learning rate 0.001, epoch 20 each 500 samples               |

---------------------------------------------------------------------
**Other branches:**
| Branch Name                                                                          | Details                                                                                                                          | Parameters                                                                 |
|--------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| RealData-12-40-MultiLayer-Loss-alpha_xx-beta_yy-lamda_zz-nn-learning_rate_rr         | DA training with the best model in RealData-12-40-Dense-WithoutNormals (12 classes). Saving model with least DA loss.            | alpha: xx, beta: yy, lamda: zz, num of sparse point: nn, learning rate: rr |
| RealData-40-MultiLayer-Loss-alpha_xx-beta_yy-lamda_zz-nn-learning_rate_rr            | DA training with the best model in RealData-40-Dense-WithoutNormals (15 classes). Saving model with best validation performance. | alpha: xx, beta: yy, lamda: zz, num of sparse point: nn, learning rate: rr |
| PointNet-(LAYER)-DA_METHOD-(Normal)-nn-alpha_xx-beta_yy-lamda_zz-learning_rate_rr    | Simulation with ModelNet10 dataset. DA training with the best model in PointNet-Dense-Normal/WithoutNormal.                      | alpha: xx, beta: yy, lamda: zz, num of sparse point: nn, learning rate: rr |
| PointNet-40-(LAYER)-DA_METHOD-(Normal)-nn-alpha_xx-beta_yy-lamda_zz-learning_rate_rr | Simulation with ModelNet40 dataset. DA training with the best model in PointNet-40-Dense-Normal/WithoutNormal.                   | alpha: xx, beta: yy, lamda: zz, num of sparse point: nn, learning rate: rr |
| Result_CF_Mat                                                                        | Generating the CF matrix using in the paper. 15 classes with 12 outputs. Unique branch, not used in other tasks.                 | ---                                                                        |
| test*                                                                                | Useless branches. Already merged into cong_dev.                                                                                  | ---                                                                        |
| RealData-40-ActiveVison*                                                             | Useless branches. The active vision training with pre-trained model from ModelNet40.                                             | ---                                                                        |
