import argparse
import os
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from . import utils
import torch.nn as nn

from active_dynamicmemory.ActiveDynamicMemory import RBACADynamicMemory
from active_dynamicmemory.ActiveDynamicMemory import MemoryItem
from abc import ABC, abstractmethod

class ActiveDynamicMemoryModel(pl.LightningModule, ABC):

    def init(self, mparams={}, modeldir=None, device=torch.device('cpu'), training=True):
        self.mparams = argparse.Namespace(**mparams)
        self.to(device)

        self.modeldir = modeldir
        
        self.mparams.seed = 42  # Replace 42 with your desired seed value

        self.model, self.stylemodel, self.gramlayers = self.load_model_stylemodel(self.mparams.droprate, load_stylemodel=training)
        if training:
            self.stylemodel.to(device)

        self.learning_rate = self.mparams.learning_rate
        self.train_counter = 0

        if self.mparams.continuous:
            self.budget = self.mparams.startbudget
            self.budgetchangecounter = 0
            if self.mparams.allowedlabelratio == 0:
                self.budgetrate = 0
            else:
                self.budgetrate = 1 / self.mparams.allowedlabelratio


        if not self.mparams.base_model is None and training:
            state_dict =  torch.load(os.path.join(modeldir, self.mparams.base_model))
            new_state_dict = {}
            for key in state_dict.keys():
                if key.startswith('model.'):
                    new_state_dict[key.replace('model.', '', 1)] = state_dict[key]
            self.model.load_state_dict(new_state_dict)

        # Initilize checkpoints to calculate BWT, FWT after training
        self.scanner_checkpoints = dict()
        self.scanner_checkpoints[self.mparams.order[0]] = True
        for scanner in self.mparams.order[1:]:
            self.scanner_checkpoints[scanner] = False

        if self.mparams.use_memory and self.mparams.continuous and training:
            self.init_memory_and_gramhooks()
            
        # print('Fim init de ADMA')

    def init_memory_and_gramhooks(self):
        self.grammatrices = []

        for layer in self.gramlayers:
            layer.register_forward_hook(self.gram_hook)

        initelements = self.getmemoryitems_from_base(num_items=self.mparams.memorymaximum)

        

        if self.mparams.method == 'RBACA':
            self.trainingsmemory = RBACADynamicMemory(initelements=initelements, 
                                                      memorymaximum=self.mparams.memorymaximum,
                                                      seed=self.mparams.seed,
                                                      perf_queue_len=self.mparams.len_perf_queue,
                                                     transformgrams=self.mparams.transformgrams,
                                                     outlier_distance=self.mparams.outlier_distance)
            self.insert_element = self.insert_element_rbaca
            # print('saiLoop2')


    def getmemoryitems_from_base(self, num_items=128):
        dl = DataLoader(self.TaskDatasetBatch(self.mparams.datasetfile,
                                            iterations=None,
                                            batch_size=self.mparams.batch_size,
                                            split=['base']),
                            batch_size=self.mparams.batch_size, num_workers=4, pin_memory=True, collate_fn=self.collate_fn)

        memoryitems = []
        self.grammatrices = []

        self.stylemodel.eval()

        for batch in dl:
            torch.cuda.empty_cache()

            x, y, scanner, filepath = batch

            if type(x) is list or type(x) is tuple:
                xstyle = torch.stack(x)
            elif x.size()[1]==1 and self.mparams.dim!=3:
                xstyle = torch.cat([x, x, x], dim=1)
            else:
                xstyle = x


            _ = self.stylemodel(xstyle.to(self.device))

            for i, f in enumerate(filepath):
                target = y[i]
                if type(target) == torch.Tensor:
                    det_target = target.detach().cpu()
                else:
                    det_target = {}
                    for k, v in target.items():
                        det_target[k] = v.detach().cpu()
                grammatrix = self.grammatrices[0][i].detach().cpu().numpy().flatten()
                memoryitems.append(MemoryItem(x[i].detach().cpu(), det_target, f, scanner[i],
                                              current_grammatrix=grammatrix,
                                              pseudo_domain=0))

            if len(memoryitems) >= num_items:
                break

            self.grammatrices = []

        return memoryitems[:num_items]

    def gram_hook(self,  m, input, output):
        if self.mparams.dim == 2:
            self.grammatrices.append(utils.gram_matrix(input[0]))
        elif self.mparams.dim == 3:
            self.grammatrices.append(utils.gram_matrix_3d(input[0]))
        else:
            raise NotImplementedError(f'gram hook with {self.mparams.dim} dimensions not defined')



    # This is called when mparams.method == 'rbaca'
    def insert_element_rbaca(self, x, y, filepath, scanner):
        budget_before = self.budget

        for i, img in enumerate(x):
            grammatrix = self.grammatrices[0][i].detach().cpu().numpy().flatten()

            target = y[i]
            if type(target) == torch.Tensor:
                det_target = target.detach().cpu()
            else:
                det_target = {}
                for k, v in target.items():
                    det_target[k] = v.detach().cpu()


            new_mi = MemoryItem(img.detach().cpu(), det_target, filepath[i], scanner[i], grammatrix)
            self.budget = self.trainingsmemory.insert_element(new_mi, self.budget, self)

        self.budget = self.trainingsmemory.check_outlier_memory(self.budget, self)
        self.trainingsmemory.counter_outlier_memory()

        if budget_before == self.budget:
            self.budgetchangecounter += 1
        else:
            self.budgetchangecounter = 1

        # form trainings X domain balanced batches to train one epoch on all newly inserted samples
        if not np.all(list(
                self.trainingsmemory.domaincomplete.values())) and self.budgetchangecounter < 10:  # only train when a domain is incomplete and new samples are inserted?

            for k, v in self.trainingsmemory.domaincomplete.items():
                # print(k)
                if not v:
                    # print('self.trainingsmemory.domainMetric[' + str(k) + '] = ' + str(self.trainingsmemory.domainMetric[k]))
                    if len(self.trainingsmemory.domainMetric[k]) == self.mparams.len_perf_queue:
                        mean_metric = np.mean(self.trainingsmemory.domainMetric[k])
                        # print(k, mean_metric)
                        if self.completed_domain(mean_metric):
                            # print('entrei if')
                            self.trainingsmemory.domaincomplete[k] = True
                            
            # print('sai loop')
            return True
        else:
            return False

    # This is called when mparams.method == 'uncertainty'
    def insert_element_uncertainty(self, x, y, filepath, scanner):
        budget_before = self.budget

        self.freeze()

        for p in self.modules():
            if isinstance(p, nn.Dropout):
                p.train()

        uncertainties = self.get_uncertainties(x)
        self.unfreeze()

        for i, img in enumerate(x):
            grammatrix = [bg[i].detach().cpu().numpy().flatten() for bg in self.grammatrices]
            target = y[i]
            if type(target) == torch.Tensor:
                det_target = target.detach().cpu()
            else:
                det_target = {}
                for k, v in target.items():
                    det_target[k] = v.detach().cpu()


            new_mi = MemoryItem(img.detach().cpu(), det_target, filepath[i], scanner[i], grammatrix[0])
            self.budget = self.trainingsmemory.insert_element(new_mi, uncertainties[i], self.budget, self)

        if budget_before == self.budget:
            self.budgetchangecounter += 1
        else:
            self.budgetchangecounter = 1

        return self.budgetchangecounter < 5

    def training_step(self, batch, batch_idx):
        x, y, scanner, filepath = batch
        
        # print('VOLTEI AO training_step')

        self.grammatrices = []

        self.stylemodel.eval()

        if self.mparams.continuous:
            # save checkpoint at scanner shift
            newshift = False
            for s in scanner:
                if s != self.mparams.order[0] and not self.scanner_checkpoints[s]:
                    newshift = True
                    shift_scanner = s
            if newshift:
                # print('new shift to', shift_scanner)
                exp_name = utils.get_expname(self.mparams)
                weights_path = self.modeldir + exp_name + '_shift_' + shift_scanner + '.pt'
                torch.save(self.model.state_dict(), weights_path)
                self.scanner_checkpoints[shift_scanner] = True
        
        # print('training_step1')

        if self.mparams.use_memory:
            #y = y[:, None]
            self.grammatrices = []
            if type(x) is list or type(x) is tuple:
                xstyle = torch.stack(x)
            elif x.size()[1] == 1 and self.mparams.dim != 3:
                xstyle = torch.cat([x, x, x], dim=1)
            else:
                xstyle = x
            _ = self.stylemodel(xstyle)

            train_a_step = self.insert_element(x, y, filepath, scanner)
            # print('train_a_step')
            # print(train_a_step)

            if train_a_step:
                x, y = self.trainingsmemory.get_training_batch(self.mparams.batch_size,
                                                                 batches=int(
                                                                     self.mparams.training_batch_size / self.mparams.batch_size))
                self.train_counter += 1
                
                # print('x')
                # # print(x)
                # print('y')
                # # print(y)
                # print('self.train_counter')
                # print(self.train_counter)
                
            else:
                return None

        loss = self.get_task_loss(x, y)
        
        self.log('train_loss', loss)
        self.grammatrices = []
        # print('loss')
        # print(loss)
        return loss
        # return None

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        if self.mparams.continuous:
            if 'randomstream' in self.mparams:
                return DataLoader(self.TaskDatasetContinuous(self.mparams.datasetfile,
                                                             transition_phase_after=self.mparams.transition_phase_after,
                                                             seed=self.mparams.seed,
                                                             order=self.mparams.order, random=True),
                                  batch_size=self.mparams.batch_size, num_workers=8, drop_last=True,
                                  collate_fn=self.collate_fn)
            else:
                return DataLoader(self.TaskDatasetContinuous(self.mparams.datasetfile,
                                                             transition_phase_after=self.mparams.transition_phase_after,
                                                             seed=self.mparams.seed,
                                                             order=self.mparams.order),
                                  batch_size=self.mparams.batch_size, num_workers=8, drop_last=True,
                                  collate_fn=self.collate_fn)
        else:
            return DataLoader(self.TaskDatasetBatch(self.mparams.datasetfile,
                                                    iterations=self.mparams.noncontinuous_steps,
                                                    batch_size=self.mparams.batch_size,
                                                    split=self.mparams.noncontinuous_train_splits,
                                                    res=self.mparams.scanner,
                                                    seed=self.mparams.seed),
                              batch_size=self.mparams.batch_size, num_workers=8, collate_fn=self.collate_fn)

    #@pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.TaskDatasetBatch(self.mparams.datasetfile,
                                                split='val', res=self.mparams.order),
                          batch_size=4,
                          num_workers=2,
                          collate_fn=self.collate_fn)




    @abstractmethod
    def load_model_stylemodel(self, droprate, load_stylemodel=False):
        pass

    @abstractmethod
    def get_task_metric(self, image, target):
        # print('tou no get_task_metric do activedynamicmemorymodel.py')
        # return 0
        pass

    @abstractmethod
    def completed_domain(self, m):
        pass

    @abstractmethod
    def get_task_loss(self, x, y):
        pass

    @abstractmethod
    def get_uncertainties(self, x):
        pass