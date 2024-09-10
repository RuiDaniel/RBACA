import torch
import torch.nn as nn
from datasets.ContinuousDataset import CardiacContinuous
from datasets.BatchDataset import CardiacBatch
from monai.metrics import DiceMetric
import monai.networks.utils as mutils

from active_dynamicmemory.ActiveDynamicMemoryModel import ActiveDynamicMemoryModel
import monai.networks.nets as monaimodels
import torchvision.models as models
import numpy as np

from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, FocalLoss

INCLUDE_BACKGROUND_TRAINING = False

LOSS_FUNCTION_NAME = 'DICE_CE' # Could be {DICE, CE, DICE_CE, FL, DICE_FL}

    
class CELoss(DiceCELoss):
    def __init__(self, include_background=True, to_onehot_y=False, sigmoid=False, softmax=False,
                 other_act=None, squared_pred=False, jaccard=False, reduction='mean',
                 smooth_nr=1e-05, smooth_dr=1e-05, batch=False, ce_weight=None, weight=None,
                 lambda_dice=1.0, lambda_ce=1.0):
        super().__init__(include_background=include_background, to_onehot_y=to_onehot_y,
                         sigmoid=sigmoid, softmax=softmax, other_act=other_act,
                         squared_pred=squared_pred, jaccard=jaccard, reduction=reduction,
                         smooth_nr=smooth_nr, smooth_dr=smooth_dr, batch=batch,
                         ce_weight=ce_weight, weight=weight, lambda_dice=lambda_dice,
                         lambda_ce=lambda_ce)

    def forward(self, input, target):
        ce_loss = super().ce(input, target)
        return ce_loss

    
class CardiacActiveDynamicMemory(ActiveDynamicMemoryModel):

    def __init__(self, mparams={}, modeldir=None, device=torch.device('cpu'), training=True):
        super(ActiveDynamicMemoryModel, self).__init__()
        self.collate_fn = None
        self.TaskDatasetBatch = CardiacBatch
        self.TaskDatasetContinuous = CardiacContinuous
        # self.loss = nn.CrossEntropyLoss()
        
        if LOSS_FUNCTION_NAME == 'DICE':
            self.loss = DiceLoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True)
        elif LOSS_FUNCTION_NAME == 'CE':
            self.loss = CELoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True)
        elif LOSS_FUNCTION_NAME == 'DICE_CE':
            self.loss = DiceCELoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True)
        elif LOSS_FUNCTION_NAME == 'DICE_FL':
            self.loss = DiceFocalLoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True)
        elif LOSS_FUNCTION_NAME == 'FL':
            self.loss = DiceFocalLoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True, lambda_dice=0.0)

        self.init(mparams=mparams, modeldir=modeldir, device=device, training=training)
        # print('Fim de CADM')

    def load_model_stylemodel(self, droprate, load_stylemodel=False):
        """
        Load the cardiac segmentation model (Res. U-Net)
        :param droprate: dropout rate to be applied
        :param load_stylemodel: If true loads the style model (needed for training)
        :return: loaded model, stylemodel, and gramlayers
        """
        model = monaimodels.UNet(
            spatial_dims=2,  # or 3 if you are working with 3D data
            in_channels=1,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            norm='batch',
            dropout=droprate,
            num_res_units=2
        )

        if load_stylemodel:
            stylemodel = models.resnet50(pretrained=True)
            gramlayers = [stylemodel.layer2[-1].conv1]
            stylemodel.eval()

            return model, stylemodel, gramlayers

        return model, None, None

    def dice_coefficient(self, prediction, target):
        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target)
        dice = (2.0 * intersection) / (union + 1e-8)  # Adding a small epsilon to avoid division by zero
        return dice

    def dice_mean_score(self, one_hot_pred, one_hot_target):
        num_classes = one_hot_pred.shape[1]  # Assuming the second dimension is the number of classes

        dice_scores = torch.zeros(num_classes, device=one_hot_pred.device)

        for i in range(num_classes):
            pred_class = one_hot_pred[:, i, ...]
            target_class = one_hot_target[:, i, ...]
            dice_scores[i] = self.dice_coefficient(pred_class, target_class)

        mean_dice = torch.mean(dice_scores)
        return mean_dice.item()

    

    def get_task_metric(self, image, target):
        """
        Task metric for cardiac segmentation is the dice score
        :param image: image the dice model should run on
        :param target: groundtruth segmentation of LV, MYO, RV
        :return: mean dice score metric
        """
        # print('tou no CARDIAC2')
        self.eval()
        x = image
        x = x[None, :].to(self.device)
        y = target[None, None, :, :].to(self.device)
        outy = self.model(x.float())
        y_hat_flat = torch.argmax(outy, dim=1)[:, None, :, :]

        one_hot_pred = mutils.one_hot(y_hat_flat, num_classes=4)
        one_hot_target = mutils.one_hot(y, num_classes=4)

        dm_m = DiceMetric(include_background=False, reduction='mean')
        
        # print('one_hot_pred')
        # print(one_hot_pred)
        # print('one_hot_target')
        # print(one_hot_target)
        # print('dm_m(one_hot_pred, one_hot_target)')
        # print(dm_m(one_hot_pred, one_hot_target))
        # print('dice_mean_score(one_hot_pred, one_hot_target)')
        # print(self.dice_mean_score(one_hot_pred, one_hot_target))
        

        # Assuming one_hot_pred and one_hot_target are already on GPU
        dice = self.dice_mean_score(one_hot_pred, one_hot_target)
        
        self.train()
        #return dice.detach().cpu().numpy()
        return dice

    def completed_domain(self, m):
        """
        Domain is completed if m larger than a threshold
        :param m: value to compare to the threshold
        :return: Wheter or not the domain is considered completed
        """
        if np.isnan(m):
            # print('apanhei o nan1')
            return False
        
        return m>self.mparams.completion_limit

    def validation_step(self, batch, batch_idx):
        """
        Validation step managed by pytorch lightning
        :param batch:
        :param batch_idx:
        :return: logs for validation
        """
        x, y, res, img = batch
        self.grammatrices = []

        y_hat = self.forward(x.float())

        res = res[0]
        
        y_toloss = y.unsqueeze(1)
        
        
        loss = self.loss(y_hat, y_toloss)
        # print('loss1', loss)
        
        
        y = y[:, None, :, :].to(self.device)
        y_hat_flat = torch.argmax(y_hat, dim=1)[:, None, :, :]


        one_hot_pred = mutils.one_hot(y_hat_flat, num_classes=4)
        one_hot_target = mutils.one_hot(y, num_classes=4)
        dm_m = DiceMetric(include_background=False, reduction='mean_batch')

        dice, *_ = dm_m(one_hot_pred, one_hot_target)

        self.log_dict({f'val_loss_{res}': loss,
                       f'val_dice_lv_{res}': dice[0],
                       f'val_dice_myo_{res}': dice[1],
                       f'val_dice_rv_{res}': dice[2]})


    def get_task_loss(self, xs, ys):
        if type(xs) is list:
            loss = None
            for i, x in enumerate(xs):
                y = ys[i]

                y = torch.stack(y).to(self.device)
                x = x.to(self.device)
                y_hat = self.forward(x.float())
                if loss is None:
                    y_toloss = y.unsqueeze(1)
                    loss = self.loss(y_hat, y_toloss)
                    # print('loss2', loss)
                else:
                    y_toloss = y.unsqueeze(1)
                    loss += self.loss(y_hat, y_toloss)
                    # print('loss3', loss)
        else:
            if type(ys) is list:
                ys = torch.stack(ys).to(self.device)
            y_hat = self.forward(xs.float())
            
            y_toloss = ys.unsqueeze(1)
            loss = self.loss(y_hat, y_toloss)
            # print('loss4', loss)
        
        # print('loss', loss)

        return loss

    def get_uncertainties(self, x):
        y_hat_flats = []
        uncertainties = []

        for i in range(self.mparams.uncertainty_iterations):
            outy = self.forward(x)
            y_hat_flat = torch.argmax(outy, dim=1)[:, None, :, :]
            y_hat_flats.append(y_hat_flat)

        for i in range(len(x)):
            y_hat_detach = [t.detach().cpu().numpy() for t in y_hat_flats]
            uncertainties.append(np.quantile(np.array(y_hat_detach).std(axis=0)[i], 0.95))

        return uncertainties

