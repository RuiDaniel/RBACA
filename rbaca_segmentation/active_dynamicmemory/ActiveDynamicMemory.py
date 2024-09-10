import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import collections
import numpy as np
from abc import ABC, abstractmethod
import random
from sklearn.metrics import mean_squared_error
import copy
import torchvision.models as models
from sklearn.cluster import KMeans
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss, FocalLoss
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

DYNAMIC_MM_ALLOCATION = False
INITIAL_SIZE = 1
PRUNING = 'EGL' # {'None', 'KMeans', 'Uncertainty', 'KU_A', 'EGL', 'DBScan', 'GMM', 'EGL_GMM'}

AL_UNCERTAINTY = False
UNCERTAINTY_THRESHOLD = 0.025


LOSS_FUNCTION_NAME = 'DICE_CE'
END = False
epsilon = 1e-8
LEARNING_RATE = 0.0001

INCLUDE_BACKGROUND_TRAINING = False



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
    

class MemoryItem():

    def __init__(self, img, target, filepath, scanner, current_grammatrix=None, pseudo_domain=None):
        self.img = img.detach().cpu()
        self.target = target
        self.filepath = filepath
        self.scanner = scanner
        self.counter = 0
        self.traincounter = 0
        self.deleteflag = False
        self.pseudo_domain = pseudo_domain
        self.current_grammatrix = current_grammatrix

class DynamicMemory(ABC):

    @abstractmethod
    def __init__(self, initelements, **kwargs):
        self.memoryfull = False
        self.memorylist = initelements
        self.memorymaximum = kwargs['memorymaximum']
        self.seed = kwargs['seed']
        self.labeling_counter = 0
        
        self.stylemodel, self.gramlayers = self.load_model_stylemodel()
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        print('device_init', self.device)
        
        self.stylemodel.to(self.device)

    @abstractmethod
    def insert_element(self, item):
        pass

    def get_training_batch(self, batchsize, force_elements=[], batches=1):
        xs = []
        ys = []

        batchsize = min(batchsize, len(self.memorylist))
        imgshape = self.memorylist[0].img.shape

        half_batch = int(batchsize / 2)

        for b in range(batches):
            j = 0
            bs = batchsize
            if len(imgshape) == 3:
                x = torch.empty(size=(batchsize, imgshape[0], imgshape[1], imgshape[2]))
            elif len(imgshape) == 4:
                x = torch.empty(size=(batchsize, imgshape[0], imgshape[1], imgshape[2], imgshape[3]))

            y = list()

            if len(force_elements) > 0:
                random.shuffle(force_elements)
                m = min(len(force_elements), half_batch)
                for mi in force_elements[-m:]:
                    if j < bs:
                        x[j] = mi.img
                        y.append(mi.target)
                        j += 1
                        mi.traincounter += 1

            bs -= j
            if bs > 0:
                random.shuffle(self.memorylist)
                for mi in self.memorylist[-bs:]:
                    x[j] = mi.img
                    y.append(mi.target)
                    j += 1
                    mi.traincounter += 1

            xs.append(x)
            ys.append(y)
        return xs, ys


class RBACADynamicMemory(DynamicMemory):

    def __init__(self, initelements, **kwargs):
        super(RBACADynamicMemory, self).__init__(initelements, **kwargs)


        self.samples_per_domain = self.memorymaximum
        self.domaincounter = {0: len(self.memorylist)} #0 is the base training domain
        self.max_per_domain = self.memorymaximum


        graminits = []
        for mi in initelements:
            graminits.append(mi.current_grammatrix)
        print('gram matrix init elements', initelements[0].current_grammatrix.shape)
        if kwargs['transformgrams']:
            self.transformer = PCA(random_state=self.seed, n_components=30)
            self.transformer.fit(graminits)
            print('fit sparse projection')
            for mi in initelements:
                mi.current_grammatrix = self.transformer.transform(mi.current_grammatrix.reshape(1, -1))
                mi.current_grammatrix = mi.current_grammatrix[0]
            trans_initelements = self.transformer.transform(graminits)
        else:
            print('no transform')
            self.transformer = None
            trans_initelements = graminits

        center = trans_initelements.mean(axis=0)
        self.centers = {0: center}
        tmp_dist = [mean_squared_error(tr, center) for tr in trans_initelements]
        self.max_center_distances = {0: np.array(tmp_dist).mean()*2}
        self.domaincomplete = {0: True}

        self.perf_queue_len = kwargs['perf_queue_len']
        self.domainMetric = {0: collections.deque(maxlen=self.perf_queue_len)}
        # print('self.domainMetric')
        # print(self.domainMetric)
        self.outlier_memory = []
        self.outlier_epochs = 15

        if 'outlier_distance' in kwargs:
            self.outlier_distance = kwargs['outlier_distance']
        else:
            self.outlier_distance = 0.20


    def check_outlier_memory(self, budget, model):
        if len(self.outlier_memory)>9 and int(budget)>=5:
            outlier_grams = [o.current_grammatrix for o in self.outlier_memory]
            distances = squareform(pdist(outlier_grams))

            distance_list = [np.array(sorted(d)[:10]).sum() for d in distances]

            # print(sorted(distance_list), sorted(distance_list)[5])

            if sorted(distance_list)[5]<self.outlier_distance:

                to_delete = np.where(np.array(distance_list) < self.outlier_distance)[0]
                new_domain_label = len(self.centers)
                self.domaincomplete[new_domain_label] = False
                self.domaincounter[new_domain_label] = 0
                self.domainMetric[new_domain_label] = collections.deque(maxlen=self.perf_queue_len)
                
#                 print('new_domain_label')
#                 print(new_domain_label)
                
#                 print('self.domainMetric[new_domain_label]')
#                 print(self.domainMetric[new_domain_label])
                
                if not DYNAMIC_MM_ALLOCATION or new_domain_label + 1 <= INITIAL_SIZE:
                    self.max_per_domain = int(self.memorymaximum/(new_domain_label+1))
                else:
                    self.memorymaximum += self.max_per_domain
                    
                    elements_to_add = self.max_per_domain
                    random_integers = random.sample(range(len(self.memorylist) + 1), min(elements_to_add, len(self.memorylist)))
                    
                    for idx, item in enumerate(self.memorylist):
                        if idx in random_integers:
                            item_aux = copy.copy(item)
                            item_aux.pseudo_domain = new_domain_label
                            item_aux.deleteflag = True
                            self.memorylist.append(item_aux)
                    
                
                print('self.max_per_domain', self.max_per_domain)
                print('no Domains', new_domain_label+1)
                print('Total size memory', len(self.memorylist))
                    
                
                self.flag_items_for_deletion(model)
                
                

                for k, d in enumerate(to_delete):
                    if int(budget)>0:
                        idx = self.find_insert_position()
                        if idx != -1:
                            elem = self.outlier_memory.pop(d-k)
                            elem.pseudo_domain = new_domain_label
                            self.memorylist[idx] = elem
                            self.domaincounter[new_domain_label] += 1
                            budget -= 1.0
                            
                            # if budget % 20 == 0:
                            #     print('budgetO', budget)
                            
                            
                            self.labeling_counter += 1
                            self.domainMetric[new_domain_label].append(model.get_task_metric(elem.img + epsilon, elem.target + epsilon))
                            # print('elem.img + eps')
                            # print(elem.img + epsilon)
                            # print('elem.target')
                            # print(elem.target + epsilon)
                            # print('model.get_task_metric(elem.img + epsilon, elem.target + epsilon)' + str(model.get_task_metric(elem.img + epsilon, elem.target + epsilon)))            
                    else:
                        if not END:
                            print('run out of budget ', budget)
                            END = True

                domain_grams = [o.current_grammatrix for o in self.get_domainitems(new_domain_label)]

                self.centers[new_domain_label] = np.array(domain_grams).mean(axis=0)
                tmp_dist = [mean_squared_error(tr, self.centers[new_domain_label]) for tr in domain_grams]
                self.max_center_distances[new_domain_label] = np.array(tmp_dist).mean() * 2


                # for elem in self.get_domainitems(new_domain_label):
                #     print('found new domain', new_domain_label, elem.scanner)

        return budget

    def find_insert_position(self):
        for idx, item in enumerate(self.memorylist):
            if item.deleteflag:
                return idx
        return -1
    
    
    def load_model_stylemodel(self):
        stylemodel = models.resnet50(pretrained=True)
        gramlayers = [stylemodel.layer2[-1].conv1]
        stylemodel.eval()

        return stylemodel, gramlayers

    
    def preprocess_with_stylemodel(self, data, stylemodel, batch_size=32):
        device = next(stylemodel.parameters()).device  # Get device of the stylemodel
        
        print('device_sm', device)
        
        num_samples = len(data)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        embeddings = []
        
        stylemodel.eval()  # Set the style model to evaluation mode
        
        with torch.no_grad():
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)
                
                batch_data = data[start_idx:end_idx]
                batch_tensor = torch.tensor(batch_data, dtype=torch.float32).to(device)
                                
                batch_tensor = batch_tensor.expand(-1, 3, -1, -1)
                                
                batch_embeddings = stylemodel(batch_tensor)
                embeddings.append(batch_embeddings)
        
        embeddings = torch.cat(embeddings, dim=0)
        
        embeddings = embeddings.cpu()
        
        return embeddings

    def dataset_distillation_k_means(self, data, labels, output_size, USE_EMBEDDINGS=True, STRATEGY=2, num_centres=5):
        if output_size == 'INF' or int(output_size) > data.shape[0]:
            distilled_data = data
            labels_distillated = labels
        else:
            if USE_EMBEDDINGS:
                # Preprocess the data using the stylemodel
                data_preprocessed = self.preprocess_with_stylemodel(data, self.stylemodel)
            else:
                data_preprocessed = data.reshape(data.shape[0], -1)
            
            # Fit K-means clustering to the preprocessed dataset
            if output_size == 'INF':
                k = data_preprocessed.shape[0]
            else:
                k = min(int(output_size), data_preprocessed.shape[0])
            
            if STRATEGY == 1:
                n_cl = k
            elif STRATEGY == 2:
                n_cl = num_centres
            else:
                n_cl = 1
                
            kmeans = KMeans(n_clusters=n_cl, random_state=self.seed)
            kmeans.fit(data_preprocessed)
            
            # Get the centroids of the clusters
            centroids = kmeans.cluster_centers_

            # Calculate the distance of each data point to the centroids
            distances = kmeans.transform(data_preprocessed)

            # Sort the data points based on their distance to the centroids
            sorted_indices = np.argsort(distances.sum(axis=1))

            if STRATEGY == 2:
                # Initialize arrays to store selected indices
                selected_indices = []
                # Loop over centroids
                
                cluster_sizes = np.bincount(kmeans.labels_)
                cluster_ids_sorted_by_size = np.argsort(cluster_sizes)
                
                remaining_size = k
                clusters_to_do = n_cl
                for i in cluster_ids_sorted_by_size:
                    # Get indices of data points assigned to current centroid
                    centroid_indices = np.where(kmeans.labels_ == i)[0]
                    # Sort centroid_indices based on distance to current centroid
                    sorted_centroid_indices = centroid_indices[np.argsort(distances[centroid_indices, i])]
                    
                    num_selected_points = min(len(sorted_centroid_indices), remaining_size // clusters_to_do)
                    print('cluster:', i, ', cluster size:', len(centroid_indices), ', remaining_size:', remaining_size, ', num_selected_points:', num_selected_points)
                    selected_indices.extend(sorted_centroid_indices[:num_selected_points])
                    remaining_size -= num_selected_points
                    clusters_to_do -= 1
                    if remaining_size <= 0 or clusters_to_do <= 0:
                        break    
                    
                # Select the data points based on selected indices
                selected_indices = np.array(selected_indices)
                distilled_data = data[selected_indices]
                labels_distillated = labels[selected_indices]
            else:
                # Select the top k data points
                distilled_data = data[sorted_indices[:k]]
                labels_distillated = labels[sorted_indices[:k]]

        return distilled_data, labels_distillated

    def dataset_distillation_k_means_old(self, data, labels, output_size, USE_EMBEDDINGS=True, STRATEGY=2, num_centres=5):
        
        if USE_EMBEDDINGS:
            # Preprocess the data using the stylemodel
            data_preprocessed = self.preprocess_with_stylemodel(data, self.stylemodel)
        else:
            data_preprocessed = data.reshape(data.shape[0], -1)
        
        # Fit K-means clustering to the preprocessed dataset
        if output_size == 'INF':
            k = data_preprocessed.shape[0]
        else:
            k = min(int(output_size), data_preprocessed.shape[0])
        
        if STRATEGY == 1:
            n_cl = k
        elif STRATEGY == 2:
            n_cl = num_centres
        else:
            n_cl = 1
            
        kmeans = KMeans(n_clusters=n_cl, random_state=self.seed)
        kmeans.fit(data_preprocessed)
        
        # Get the centroids of the clusters
        centroids = kmeans.cluster_centers_

        # Calculate the distance of each data point to the centroids
        distances = kmeans.transform(data_preprocessed)

        # Sort the data points based on their distance to the centroids
        sorted_indices = np.argsort(distances.sum(axis=1))

        if STRATEGY == 2:
            # Initialize arrays to store selected indices
            selected_indices = []
            # Loop over centroids
            for i in range(n_cl):
                # Get indices of data points assigned to current centroid
                centroid_indices = np.where(kmeans.labels_ == i)[0]
                # Sort centroid_indices based on distance to current centroid
                sorted_centroid_indices = centroid_indices[np.argsort(distances[centroid_indices, i])]
                # Select top k/num_centres points from sorted indices
                selected_indices.extend(sorted_centroid_indices[:k // num_centres])
            # Select the data points based on selected indices
            selected_indices = np.array(selected_indices)
            distilled_data = data[selected_indices]
            labels_distillated = labels[selected_indices]
        else:
            # Select the top k data points
            distilled_data = data[sorted_indices[:k]]
            labels_distillated = labels[sorted_indices[:k]]

        return distilled_data, labels_distillated

    def dataset_distillation_egl(self, data, labels, output_size, model, batch_size=32):
        if output_size == 'INF' or int(output_size) > data.shape[0]:
            distilled_data = data
            labels_distillated = labels
        else:
            data_aux = torch.tensor(data, dtype=torch.float32).to(self.device)
            labels_aux = torch.tensor(labels, dtype=torch.long).to(self.device)  

            if LOSS_FUNCTION_NAME == 'DICE':
                loss_function = DiceLoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True)
            elif LOSS_FUNCTION_NAME == 'CE':
                loss_function = CELoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True)
            elif LOSS_FUNCTION_NAME == 'DICE_CE':
                loss_function = DiceCELoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True)
            elif LOSS_FUNCTION_NAME == 'DICE_FL':
                loss_function = DiceFocalLoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True)
            elif LOSS_FUNCTION_NAME == 'FL':
                loss_function = DiceFocalLoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True, lambda_dice=0.0)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
            train_dataset = torch.utils.data.TensorDataset(data_aux, labels_aux)
            
            
            model.train()
            
            # Calculate gradients and their lengths for each sample in the training set
            gradient_lengths = []

            for d, t in train_dataset:
                d, t = d.to(self.device), t.to(self.device)
                d = d.unsqueeze(0)
                t = t.unsqueeze(0)
                d.requires_grad = True  # Enable gradient tracking
                optimizer.zero_grad()
                output = model.forward(d)
                t = t.unsqueeze(1)
                loss = loss_function(output, t)
                gradients = torch.autograd.grad(loss, d, retain_graph=True)[0]
                gradient_lengths.append(gradients.norm().item())  # Compute the L2 norm of the gradient

            # print('gradient_lengths', gradient_lengths)
            # Sort samples based on gradient lengths
            sorted_indices = sorted(range(len(gradient_lengths)), key=lambda i: gradient_lengths[i], reverse=True)
                
            # Select top samples based on gradient lengths
            distilled_data = data[sorted_indices[:output_size]]
            labels_distillated = labels[sorted_indices[:output_size]]

        return distilled_data, labels_distillated

    def compute_segmentation_entropy(self, segmentation_probs):
        segmentation_probs_cpu = segmentation_probs.cpu()  # Move tensor to CPU
        num_images, num_classes, height, width = segmentation_probs_cpu.shape
        
        # print('num_images, num_classes, height, width', num_images, num_classes, height, width)
        
        # Reshape segmentation_probs to have pixels as the last dimension
        reshaped_probs = segmentation_probs_cpu.permute(0, 2, 3, 1).reshape(-1, num_classes)

        # print('reshaped_probs_before', reshaped_probs)
        
        reshaped_probs += 1e-15
        
        # print('reshaped_probs_after', reshaped_probs)
        
        # Calculate entropy for all pixels in all images simultaneously
        pixel_entropy = -torch.sum(reshaped_probs * torch.log(reshaped_probs), dim=1)
        # print('pixel_entropy', pixel_entropy)
        
        # Reshape pixel_entropy to have one value per image
        entropies = pixel_entropy.view(num_images, height * width).mean(dim=1)

        return entropies.numpy()  # Convert result to NumPy array


    def get_segmentation_probs(self, model, input_data):
        # Set model to evaluation mode
        model.eval()

        # Forward pass
        with torch.no_grad():
            input_data = torch.tensor(input_data, dtype=torch.float32).to(self.device)
            output_tensor = model(input_data)

        # Apply softmax activation to convert logits to probabilities
        segmentation_probs = F.softmax(output_tensor, dim=1)

        return segmentation_probs

    def dataset_distillation_eglgmm(self, data, labels, output_size, model, batch_size=32, num_centres=5, USE_EMBEDDINGS=True):
        if output_size == 'INF' or int(output_size) > data.shape[0]:
            distilled_data = data
            labels_distillated = labels
        else:
            stylemodel, gramlayers = self.load_model_stylemodel()
            stylemodel.to(self.device)

            if USE_EMBEDDINGS:
                # Preprocess the data using the stylemodel
                data_preprocessed = self.preprocess_with_stylemodel(data, stylemodel)
            else:
                data_preprocessed = data.reshape(data.shape[0], -1)
                
            print('data_to_distillate.shape', data_preprocessed.shape)
            
            # Fit Gaussian Mixture Model clustering to the preprocessed dataset

            n_components = num_centres

            gmm = GaussianMixture(n_components=n_components, random_state=self.seed)
            gmm.fit(data_preprocessed)
            
            # Get the cluster assignments and responsibilities of each data point to the clusters
            cluster_labels = gmm.predict(data_preprocessed)
            responsibilities = gmm.predict_proba(data_preprocessed)
            
            # Calculate the number of data points in each cluster
            cluster_sizes = np.bincount(cluster_labels)
            cluster_ids_sorted_by_size = np.argsort(cluster_sizes)
            
            # Initialize arrays to store selected indexes
            selected_indexes = []
            remaining_size = output_size
            clusters_to_do = len(cluster_ids_sorted_by_size)
            

            if LOSS_FUNCTION_NAME == 'DICE':
                loss_function = DiceLoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True)
            elif LOSS_FUNCTION_NAME == 'CE':
                loss_function = CELoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True)
            elif LOSS_FUNCTION_NAME == 'DICE_CE':
                loss_function = DiceCELoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True)
            elif LOSS_FUNCTION_NAME == 'DICE_FL':
                loss_function = DiceFocalLoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True)
            elif LOSS_FUNCTION_NAME == 'FL':
                loss_function = DiceFocalLoss(include_background=INCLUDE_BACKGROUND_TRAINING, to_onehot_y=True, softmax=True, lambda_dice=0.0)

            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)   
            
            distilled_data = []
            labels_distillated = []
            data_excodes_distillated = []
            labels_excodes_distillated = []    
            
            # Loop over clusters sorted by size
            for cluster_id in cluster_ids_sorted_by_size:
                selected_indexes = []
                distilled_data1 = []
                labels_distillated1 = []
                data_excodes_distillated1 = []
                labels_excodes_distillated1 = []
                distilled_data2 = []
                labels_distillated2 = []
                data_excodes_distillated2 = []
                labels_excodes_distillated2 = []   
            
                
                # Get indexes of data points assigned to the current cluster
                cluster_indexes = np.where(cluster_labels == cluster_id)[0]
                
                # Select top k/num_centres points from sorted indexes
                num_selected_points = min(len(cluster_indexes), remaining_size // clusters_to_do)
                
                if num_selected_points == len(cluster_indexes):
                    num_selected_points_gmm = num_selected_points
                    num_selected_points_egl = 0
                else:
                    num_selected_points_gmm = int(num_selected_points / 2)
                    num_selected_points_egl = num_selected_points - num_selected_points_gmm
                                
                # Sort cluster indexes based on the responsibilities of the data points in the cluster
                sorted_cluster_indexes = cluster_indexes[np.argsort(responsibilities[cluster_indexes, cluster_id])]
                
                if num_selected_points_gmm > 0:
                    selected_indexes.extend(sorted_cluster_indexes[:num_selected_points_gmm])
                    distilled_data1 = data[selected_indexes]
                    labels_distillated1 = labels[selected_indexes]
                    
                    
                all_indexes = set(cluster_indexes)
                selected_indexes = np.array(selected_indexes)
                selected_set = set(selected_indexes)
                unselected_indexes = all_indexes - selected_set
                unselected_indexes = np.array(list(unselected_indexes))
                
                    
                
                if num_selected_points_egl > 0 and len(unselected_indexes) > 0:
                    remaining_data = data[unselected_indexes]
                    remaining_labels = labels[unselected_indexes]
                    
                    data_aux = torch.tensor(remaining_data, dtype=torch.float32).to(self.device)
                    labels_aux = torch.tensor(remaining_labels, dtype=torch.long).to(self.device)  
                    train_dataset = torch.utils.data.TensorDataset(data_aux, labels_aux)
                            
                    model.train()
                    
                    # Calculate gradients and their lengths for each sample in the training set
                    gradient_lengths = []
                    for d, t in train_dataset:
                        d, t = d.to(self.device), t.to(self.device)
                        d = d.unsqueeze(0)
                        t = t.unsqueeze(0)
                        d.requires_grad = True  # Enable gradient tracking
                        optimizer.zero_grad()
                        output = model.forward(d)
                        t = t.unsqueeze(1)
                        loss = loss_function(output, t)
                        gradients = torch.autograd.grad(loss, d, retain_graph=True)[0]
                        gradient_lengths.append(gradients.norm().item())  # Compute the L2 norm of the gradient

                    # Sort samples based on gradient lengths
                    sorted_indexes = sorted(range(len(gradient_lengths)), key=lambda i: gradient_lengths[i], reverse=True)
                    
                    selected_indexes = sorted_indexes[:num_selected_points_egl]
                    distilled_data2 = remaining_data[selected_indexes]
                    labels_distillated2 = remaining_labels[selected_indexes]
                
                if len(distilled_data1) > 0:
                    distilled_data.extend(np.array(distilled_data1))
                    labels_distillated.extend(np.array(labels_distillated1))
                    data_excodes_distillated.extend(np.array(data_excodes_distillated1))
                    labels_excodes_distillated.extend(np.array(labels_excodes_distillated1))
                
                if len(distilled_data2) > 0:
                    distilled_data.extend(np.array(distilled_data2))
                    labels_distillated.extend(np.array(labels_distillated2))
                    data_excodes_distillated.extend(np.array(data_excodes_distillated2))
                    labels_excodes_distillated.extend(np.array(labels_excodes_distillated2))
                
                num_selected_points = len(distilled_data1) + len(distilled_data2)
                print('cluster:', cluster_id, ', cluster size:', len(cluster_indexes), ', remaining_size:', remaining_size, 'num_selected_points:, ', num_selected_points)

                    
                remaining_size -= num_selected_points
                clusters_to_do -= 1
                if remaining_size <= 0 or clusters_to_do <= 0:
                    break

        return distilled_data, labels_distillated

    def dataset_distillation_uncertainty(self, data, labels, output_size, model, batch_size=32):
        if output_size == 'INF' or int(output_size) > data.shape[0]:
            distilled_data = data
            labels_distillated = labels
        else:
            num_samples = len(data)
            uncertainties = []
        
            for i in range(0, num_samples, batch_size):
                data_batch = data[i:i+batch_size]
                
                segmentation_probs_batch = self.get_segmentation_probs(model, data_batch)
                uncertainties_batch = self.compute_segmentation_entropy(segmentation_probs_batch)
                
                for u in uncertainties_batch:
                    uncertainties.append(u)
            
            sorted_indices = np.argsort(uncertainties)
            
            distilled_data = data[sorted_indices[:output_size]]
            labels_distillated = labels[sorted_indices[:output_size]]

        return distilled_data, labels_distillated
    
    def dataset_distillation_k_means_uncertainty_a(self, data, labels, output_size, model, batch_size=32, USE_EMBEDDINGS=True, STRATEGY=2, num_centres=5):
        if output_size == 'INF' or int(output_size) > data.shape[0]:
            distilled_data = data
            labels_distillated = labels
        else:
            
            if USE_EMBEDDINGS:
                # Preprocess the data using the stylemodel
                data_preprocessed = self.preprocess_with_stylemodel(data, self.stylemodel)
            else:
                data_preprocessed = data.reshape(data.shape[0], -1)
            
            output_size = min(int(output_size), data_preprocessed.shape[0])
            # output_size_kmeans = int(output_size / 2)
            # output_size_uncertainty = int(output_size - output_size_kmeans)
            
            n_cl = num_centres
            
            kmeans = KMeans(n_clusters=n_cl, random_state=self.seed)
            kmeans.fit(data_preprocessed)
            
            # Get the centroids of the clusters
            centroids = kmeans.cluster_centers_

            # Calculate the distance of each data point to the centroids
            distances = kmeans.transform(data_preprocessed)

            # Sort the data points based on their distance to the centroids
            # sorted_indexes = np.argsort(distances.sum(axis=1))

        
            distilled_data = []
            labels_distillated = []
            
            cluster_sizes = np.bincount(kmeans.labels_)
            cluster_ids_sorted_by_size = np.argsort(cluster_sizes)
            
            remaining_size = output_size
            clusters_to_do = n_cl
                
            # Loop over centroids
            for i in cluster_ids_sorted_by_size:
                selected_indexes = []
                
                aux_num_selected = 0
                
                distilled_data1 = []
                labels_distillated1 = []
                distilled_data2 = []
                labels_distillated2 = []
                
                # Get indexes of data points assigned to current centroid
                centroid_indexes = np.where(kmeans.labels_ == i)[0]
                # Sort centroid_indexes based on distance to current centroid
                sorted_centroid_indexes = centroid_indexes[np.argsort(distances[centroid_indexes, i])]
                
                num_selected_points = min(len(sorted_centroid_indexes), remaining_size // clusters_to_do)
                
                
                to_select_1 = int(num_selected_points / 2)
                to_select_2 = num_selected_points - to_select_1
                
                
                selected_indexes.extend(sorted_centroid_indexes[:to_select_1])
                
                
                # Select the data points based on selected indexes
                selected_indexes = np.array(selected_indexes)
                if len(selected_indexes) > 0:
                    
                    distilled_data1 = data[selected_indexes]
                    labels_distillated1 = labels[selected_indexes]
                
                all_indexes = set(centroid_indexes)
                selected_set = set(selected_indexes)
                unselected_indexes = all_indexes - selected_set
                unselected_indexes = np.array(list(unselected_indexes))
                
                to_select = min(to_select_2, len(unselected_indexes)) 
                
                if to_select > 0:
                    remaining_data = data[unselected_indexes]
                    remaining_labels = labels[unselected_indexes]
                
                    num_samples = len(remaining_data)
                    uncertainties = []
                
                    for i in range(0, num_samples, batch_size):
                        data_batch = remaining_data[i:i+batch_size]
                        
                        segmentation_probs_batch = self.get_segmentation_probs(model, data_batch)
                        uncertainties_batch = self.compute_segmentation_entropy(segmentation_probs_batch)
                        
                        for u in uncertainties_batch:
                            uncertainties.append(u)
                    
                    sorted_indexes = np.argsort(uncertainties)
                    selected_indexes = sorted_indexes[:to_select]
                    
                    distilled_data2 = remaining_data[selected_indexes]
                    labels_distillated2 = remaining_labels[selected_indexes]

                if len(distilled_data1) > 0:
                    distilled_data.extend(np.array(distilled_data1))
                    labels_distillated.extend(np.array(labels_distillated1))
                
                if len(distilled_data2) > 0:
                    distilled_data.extend(np.array(distilled_data2))
                    labels_distillated.extend(np.array(labels_distillated2))
                
                aux_num_selected += len(distilled_data1) + len(distilled_data2)
                    
                print('cluster:', i, ', cluster size:', len(centroid_indexes), ', remaining_size:', remaining_size, ', num_selected_points:', aux_num_selected)
                remaining_size = output_size - len(distilled_data)
                clusters_to_do -= 1
                if remaining_size <= 0 or clusters_to_do <= 0:
                    break  
                

        return distilled_data, labels_distillated

    def dataset_distillation_k_means_uncertainty_a_old(self, data, labels, output_size, model, batch_size=32, USE_EMBEDDINGS=True, STRATEGY=2, num_centres=5):
        if output_size == 'INF' or int(output_size) > data.shape[0]:
            distilled_data = data
            labels_distillated = labels
        else:
            
            if USE_EMBEDDINGS:
                # Preprocess the data using the stylemodel
                data_preprocessed = self.preprocess_with_stylemodel(data, self.stylemodel)
            else:
                data_preprocessed = data.reshape(data.shape[0], -1)
            
            output_size = min(int(output_size), data_preprocessed.shape[0])
            output_size_kmeans = int(output_size / 2)
            output_size_uncertainty = int(output_size - output_size_kmeans)
            
            n_cl = num_centres
            
            kmeans = KMeans(n_clusters=n_cl, random_state=self.seed)
            kmeans.fit(data_preprocessed)
            
            # Get the centroids of the clusters
            centroids = kmeans.cluster_centers_

            # Calculate the distance of each data point to the centroids
            distances = kmeans.transform(data_preprocessed)

            # Sort the data points based on their distance to the centroids
            # sorted_indexes = np.argsort(distances.sum(axis=1))

        
            distilled_data = []
            labels_distillated = []
            # Loop over centroids
            for i in range(n_cl):
                selected_indexes = []
                distilled_data1 = []
                distilled_data2 = []
                # Get indexes of data points assigned to current centroid
                centroid_indexes = np.where(kmeans.labels_ == i)[0]
                
                to_select = min(output_size_kmeans // num_centres, len(centroid_indexes))
                
                # Sort centroid_indexes based on distance to current centroid
                sorted_centroid_indexes = centroid_indexes[np.argsort(distances[centroid_indexes, i])]
                # Select top k/num_centres points from sorted indexes
                selected_indexes = sorted_centroid_indexes[:to_select]
                
                # Select the data points based on selected indexes
                selected_indexes = np.array(selected_indexes)
                if len(selected_indexes) > 0:
                    distilled_data1 = data[selected_indexes]
                    labels_distillated1 = labels[selected_indexes]
                
                all_indexes = set(centroid_indexes)
                selected_set = set(selected_indexes)
                unselected_indexes = all_indexes - selected_set
                unselected_indexes = np.array(list(unselected_indexes))
                
                to_select = min(output_size_uncertainty // num_centres, len(unselected_indexes)) 
                
                if to_select > 0:
                    remaining_data = data[unselected_indexes]
                    remaining_labels = labels[unselected_indexes]
                
                    num_samples = len(remaining_data)
                    uncertainties = []
                
                    for i in range(0, num_samples, batch_size):
                        data_batch = remaining_data[i:i+batch_size]
                        
                        segmentation_probs_batch = self.get_segmentation_probs(model, data_batch)
                        uncertainties_batch = self.compute_segmentation_entropy(segmentation_probs_batch)
                        
                        for u in uncertainties_batch:
                            uncertainties.append(u)
                    
                    sorted_indexes = np.argsort(uncertainties)
                    selected_indexes = sorted_indexes[:to_select]
                    
                    distilled_data2 = remaining_data[selected_indexes]
                    labels_distillated2 = remaining_labels[selected_indexes]

                if len(distilled_data1) > 0:
                    distilled_data.extend(np.array(distilled_data1))
                    labels_distillated.extend(np.array(labels_distillated1))
                
                if len(distilled_data2) > 0:
                    distilled_data.extend(np.array(distilled_data2))
                    labels_distillated.extend(np.array(labels_distillated2))

        return distilled_data, labels_distillated
    
    def dataset_distillation_k_means_uncertainty_b(self, data, labels, output_size, model, batch_size=32, USE_EMBEDDINGS=True, STRATEGY=2, num_centres=5):
        if output_size == 'INF' or int(output_size) > data.shape[0]:
            distilled_data = data
            labels_distillated = labels
        else:
            
            if USE_EMBEDDINGS:
                # Preprocess the data using the stylemodel
                data_preprocessed = self.preprocess_with_stylemodel(data, self.stylemodel)
            else:
                data_preprocessed = data.reshape(data.shape[0], -1)
            
            output_size_kmeans = int(output_size / 2)
            output_size_uncertainty = int(output_size - output_size_kmeans)
            
            k = min(int(output_size_kmeans), data_preprocessed.shape[0])
            
            if STRATEGY == 1:
                n_cl = k
            elif STRATEGY == 2:
                n_cl = num_centres
            else:
                n_cl = 1
                
            kmeans = KMeans(n_clusters=n_cl, random_state=self.seed)
            kmeans.fit(data_preprocessed)
            
            # Get the centroids of the clusters
            centroids = kmeans.cluster_centers_

            # Calculate the distance of each data point to the centroids
            distances = kmeans.transform(data_preprocessed)

            # Sort the data points based on their distance to the centroids
            sorted_indices = np.argsort(distances.sum(axis=1))

            if STRATEGY == 2:
                # Initialize arrays to store selected indices
                selected_indices = []
                # Loop over centroids
                for i in range(n_cl):
                    # Get indices of data points assigned to current centroid
                    centroid_indices = np.where(kmeans.labels_ == i)[0]
                    # Sort centroid_indices based on distance to current centroid
                    sorted_centroid_indices = centroid_indices[np.argsort(distances[centroid_indices, i])]
                    # Select top k/num_centres points from sorted indices
                    selected_indices.extend(sorted_centroid_indices[:k // num_centres])
                # Select the data points based on selected indices
                selected_indices = np.array(selected_indices)
                distilled_data1 = data[selected_indices]
                labels_distillated1 = labels[selected_indices]
                
                total_indices = len(labels)
                all_indices = set(range(total_indices))
                selected_set = set(selected_indices)
                unselected_indices = all_indices - selected_set
                unselected_indices = np.array(list(unselected_indices))

                remaining_data = data[unselected_indices]
                remaining_labels = labels[unselected_indices]
            else:
                # Select the top k data points
                distilled_data1 = data[sorted_indices[:output_size_kmeans]]
                labels_distillated1 = labels[sorted_indices[:output_size_kmeans]]
                remaining_data = data[sorted_indices[output_size_kmeans:]]
                remaining_labels = labels[sorted_indices[output_size_kmeans:]]
    
            num_samples = len(remaining_data)
            uncertainties = []
        
            for i in range(0, num_samples, batch_size):
                data_batch = remaining_data[i:i+batch_size]
                
                segmentation_probs_batch = self.get_segmentation_probs(model, data_batch)
                uncertainties_batch = self.compute_segmentation_entropy(segmentation_probs_batch)
                
                for u in uncertainties_batch:
                    uncertainties.append(u)
            
            sorted_indices = np.argsort(uncertainties)
            
            distilled_data2 = remaining_data[sorted_indices[:output_size_uncertainty]]
            labels_distillated2 = remaining_labels[sorted_indices[:output_size_uncertainty]]

            distilled_data = np.concatenate((distilled_data1, distilled_data2), axis=0)
            labels_distillated = np.concatenate((labels_distillated1, labels_distillated2), axis=0)

        return distilled_data, labels_distillated

    def dataset_distillation_db_scan(self, data, labels, output_size, USE_EMBEDDINGS=True, eps=0.1, min_samples=3):
        if output_size == 'INF' or int(output_size) > data.shape[0]:
            distilled_data = data
            labels_distillated = labels
        else:
            # Preprocess the data if necessary
            if USE_EMBEDDINGS:
                data_preprocessed = self.preprocess_with_stylemodel(data, self.stylemodel)
            else:
                data_preprocessed = data.reshape(data.shape[0], -1)
                
            print('data_to_distillate.shape', data_preprocessed.shape)
            target_core_points = max(0.3 * len(data_preprocessed), output_size)
            # print('target_core_points', target_core_points)
            
            
            num_iter = 0
            max_iter = 60
            min_search_clusters = 4
            max_search_clusters = 21
            # # Attempt to fit DBSCAN to the preprocessed dataset
            while True:
                num_iter += 1
                # Initialize DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            
                dbscan.fit(data_preprocessed)
                
                # Get the cluster labels assigned by DBSCAN
                cluster_labels = dbscan.labels_
                
                # Number of clusters found by DBSCAN (excluding noise)
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                
                num_core_points = np.sum(cluster_labels != -1)
                
                # If at least one cluster is found, break the loop
                if num_core_points >= target_core_points and n_clusters >= min_search_clusters and n_clusters <= max_search_clusters:
                    break
                
                # If no clusters are found, reduce eps and try again
                eps *= 1.1  # You can adjust the reduction factor as needed
                
                
                if num_iter == max_iter:
                    num_iter = 0
                    if min_samples < 15:
                        min_samples += 1
                        print('min_samples', min_samples)
                    elif min_search_clusters == 4 and max_search_clusters == 21:
                        min_search_clusters = 1
                        max_search_clusters = min(output_size, 500)
                        min_samples = 3
                        eps = 0.1
                        print('retirei maximos')
                    else:
                        print('ERROR, NO FINE-TUNING FOUND!')
                        distilled_data = data[0:output_size]
                        labels_distillated = labels[0:output_size]
                        return
                    
            
            print('n_clusters', n_clusters)
            print('eps', eps)
            # input()
            # Initialize arrays to store selected indexes
            selected_indexes = []
            remaining_size = output_size
            
            # Initialize an empty list to store cluster IDs and their corresponding valid_indexes length
            cluster_valid_indexes_lengths = []

            # Loop over clusters to calculate valid_indexes length for each cluster
            for cluster_id in range(n_clusters):
                cluster_indexes = np.where(cluster_labels == cluster_id)[0]
                valid_indexes = np.intersect1d(cluster_indexes, dbscan.core_sample_indices_)
                cluster_valid_indexes_lengths.append((cluster_id, len(valid_indexes)))

            # Sort clusters by the length of valid_indexes
            cluster_valid_indexes_lengths.sort(key=lambda x: x[1])

            # print('cluster_valid_indexes_lengths', cluster_valid_indexes_lengths)

            clusters_to_do = len(cluster_valid_indexes_lengths)
            # Loop over sorted clusters
            for cluster_id, _ in cluster_valid_indexes_lengths:
                # Get indexes of data points belonging to the current cluster
                cluster_indexes = np.where(cluster_labels == cluster_id)[0]

                # print('cluster_indexes.shape', cluster_indexes.shape)
                # print('dbscan.core_sample_indices_.shape', dbscan.core_sample_indices_.shape)
                # print('dbscan.components_.shape', dbscan.components_.shape)
                
                # print('cluster_indexes', cluster_indexes)
                # print('dbscan.core_sample_indices_', dbscan.core_sample_indices_)

                # Filter out indexes that are out of bounds
                valid_indexes = np.intersect1d(cluster_indexes, dbscan.core_sample_indices_)

                # Check if there are any valid indexes left
                if len(valid_indexes) == 0:
                    continue
                
                # print('valid_indexes', valid_indexes)
                # print('min(cluster_indexes)', min(cluster_indexes))
                # print('valid_indexes - min(cluster_indexes)', valid_indexes - min(cluster_indexes))

                
                # Calculate distances of each data point to the cluster core
                # core_distances = dbscan.components_[valid_indexes - min(cluster_indexes), :]
                
                indxs = np.where(np.isin(dbscan.core_sample_indices_, valid_indexes))[0]
                
                # print('indxs.shape', indxs.shape)
                # print('indxs', indxs)
                
                core_distances = dbscan.components_[indxs, :]
                
                distances_to_core = np.linalg.norm(core_distances - core_distances.mean(axis=0), axis=1)

                # Sort cluster_indexes based on distance to cluster core
                sorted_cluster_indexes = valid_indexes[np.argsort(distances_to_core)]

                # Select top k/num_centres points from sorted indexes
                num_selected_points = min(len(sorted_cluster_indexes), remaining_size // clusters_to_do)
                
                if num_selected_points == 0 and remaining_size > 0:
                    num_selected_points = 1
                
                print('cluster:', cluster_id, ', cluster size:', len(sorted_cluster_indexes), ', remaining_size:', remaining_size, 'num_selected_points:, ', num_selected_points)
                selected_indexes_list = list(selected_indexes)
                selected_indexes_list.extend(sorted_cluster_indexes[:num_selected_points])
                selected_indexes = np.array(selected_indexes_list)
            
                remaining_size -= num_selected_points
                clusters_to_do -= 1
                
            # Select the data points based on selected indexes
            selected_indexes = np.array(selected_indexes)
            distilled_data = data[selected_indexes]
            labels_distillated = labels[selected_indexes]
            
            print('distilled_data.shape', distilled_data.shape)
            # input()

        return distilled_data, labels_distillated


    def dataset_distillation_gmm(self, data, labels, output_size, USE_EMBEDDINGS=True, num_centres=5):
    
        if output_size == 'INF' or int(output_size) > data.shape[0]:
            distilled_data = data
            labels_distillated = labels
        else:
            if USE_EMBEDDINGS:
                # Preprocess the data using the stylemodel
                data_preprocessed = self.preprocess_with_stylemodel(data, self.stylemodel)
            else:
                data_preprocessed = data.reshape(data.shape[0], -1)
                
            print('data_to_distillate.shape', data_preprocessed.shape)
            
            # Fit Gaussian Mixture Model clustering to the preprocessed dataset

            n_components = num_centres

            gmm = GaussianMixture(n_components=n_components, random_state=self.seed)
            gmm.fit(data_preprocessed)
            
            # Get the cluster assignments and responsibilities of each data point to the clusters
            cluster_labels = gmm.predict(data_preprocessed)
            responsibilities = gmm.predict_proba(data_preprocessed)
            
            # Calculate the number of data points in each cluster
            cluster_sizes = np.bincount(cluster_labels)
            cluster_ids_sorted_by_size = np.argsort(cluster_sizes)
            
            # Initialize arrays to store selected indices
            selected_indices = []
            remaining_size = output_size
            clusters_to_do = len(cluster_ids_sorted_by_size)
            
            # Loop over clusters sorted by size
            for cluster_id in cluster_ids_sorted_by_size:
                # Get indices of data points assigned to the current cluster
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                # Sort cluster indices based on the responsibilities of the data points in the cluster
                sorted_cluster_indices = cluster_indices[np.argsort(responsibilities[cluster_indices, cluster_id])]
                # Select top k/num_centres points from sorted indices
                num_selected_points = min(len(sorted_cluster_indices), remaining_size // clusters_to_do)
                # print('cluster:', cluster_id, ', cluster size:', len(cluster_indices), ', remaining_size:', remaining_size, 'num_selected_points:, ', num_selected_points)
                selected_indices.extend(sorted_cluster_indices[:num_selected_points])
                remaining_size -= num_selected_points
                clusters_to_do -= 1
                if remaining_size <= 0 or clusters_to_do <= 0:
                    break
            
            # Select the data points based on selected indices
            selected_indices = np.array(selected_indices)
            distilled_data = data[selected_indices]
            labels_distillated = labels[selected_indices]
            
        return distilled_data, labels_distillated

    def flag_items_for_deletion(self, model):
        for k, v in self.domaincomplete.items():
            domain_items = self.get_domainitems(k)
            domain_count = len(domain_items)
            for di in domain_items:
                if di.deleteflag:
                    domain_count-= 1
            if domain_count>self.max_per_domain:
                todelete = domain_count-self.max_per_domain
                
                if PRUNING == 'None':
                    for item in self.memorylist:
                        if todelete>0:
                            if item.pseudo_domain==k:
                                if not item.deleteflag:
                                    item.deleteflag = True
                                    todelete -= 1
                
                else:
                    to_delete_initial = todelete
                    print('to_delete_initial', to_delete_initial)
                    domain_items_valid = [item for item in domain_items if not item.deleteflag]
                    
                    print('len(domain_items_valid)', len(domain_items_valid))
                    
                    domain_items_valid_data = np.array([item.img for item in domain_items_valid])
                    
                    domain_items_valid_labels = np.array([item.target for item in domain_items_valid]     )
                    
                    if PRUNING == 'KMeans':
                        domain_items_selected_data, _ = self.dataset_distillation_k_means(domain_items_valid_data, domain_items_valid_labels, len(domain_items_valid) - todelete)
                    elif PRUNING == 'Uncertainty':
                        domain_items_selected_data, _ = self.dataset_distillation_uncertainty(domain_items_valid_data, domain_items_valid_labels, len(domain_items_valid) - todelete, model)
                    elif PRUNING == 'EGL':
                        domain_items_selected_data, _ = self.dataset_distillation_egl(domain_items_valid_data, domain_items_valid_labels, len(domain_items_valid) - todelete, model)
                    elif PRUNING == 'KU_A':
                        domain_items_selected_data, _ = self.dataset_distillation_k_means_uncertainty_a(domain_items_valid_data, domain_items_valid_labels, len(domain_items_valid) - todelete, model)
                    elif PRUNING == 'DBScan':
                        domain_items_selected_data, _ = self.dataset_distillation_db_scan(domain_items_valid_data, domain_items_valid_labels, len(domain_items_valid) - todelete)
                    elif PRUNING == 'KU_B':
                        domain_items_selected_data, _ = self.dataset_distillation_k_means_uncertainty_b(domain_items_valid_data, domain_items_valid_labels, len(domain_items_valid) - todelete, model)
                    elif PRUNING == 'GMM':
                        domain_items_selected_data, _ = self.dataset_distillation_gmm(domain_items_valid_data, domain_items_valid_labels, len(domain_items_valid) - todelete)
                    elif PRUNING == 'EGL_GMM':
                        domain_items_selected_data, _ = self.dataset_distillation_eglgmm(domain_items_valid_data, domain_items_valid_labels, len(domain_items_valid) - todelete, model)
                    else:
                        print('ERROR! Invalid Smart Cut')
                    
                    print('len(domain_items_selected_data)', len(domain_items_selected_data)) 
                    
                    for di in domain_items_valid:
                        img_tuple = tuple(di.img.flatten().tolist())  # Convert the img tensor to a tuple of its flattened values
                        img_present = any(img_tuple == tuple(item.flatten().tolist()) for item in domain_items_selected_data)
                        if not img_present:
                            di.deleteflag = True
                            todelete -= 1

                    
                    to_delete_final = todelete
                    print('to_delete_final', to_delete_final)
                    
                    if to_delete_final > 0:  
                        print('Error, to_delete_final > 0')
                        
                    domain_items_valid_final = [item for item in domain_items if not item.deleteflag]
                    
                    print('len(domain_items_valid_final)', len(domain_items_valid_final))
                    # input()


    def counter_outlier_memory(self):
        for item in self.outlier_memory:
            item.counter += 1
            if item.counter>self.outlier_epochs:
                self.outlier_memory.remove(item)

    def insert_element(self, item, budget, model):
        if self.transformer is not None:
            item.current_grammatrix = self.transformer.transform(item.current_grammatrix.reshape(1, -1))
            item.current_grammatrix = item.current_grammatrix[0]

        domain = self.check_pseudodomain(item.current_grammatrix)
        item.pseudo_domain = domain

        if domain==-1:
            #insert into outlier memory
            #check outlier memory for new clusters
            self.outlier_memory.append(item)
        else:
            if int(budget)>0:
                if not AL_UNCERTAINTY:
                    add_element = not self.domaincomplete[domain]
                else:
                    data_batch = (item.img).unsqueeze(0)
                    segmentation_probs = self.get_segmentation_probs(model, data_batch)
                    uncertainty = self.compute_segmentation_entropy(segmentation_probs)
                    add_element = uncertainty > UNCERTAINTY_THRESHOLD
                    
                if add_element:                
                    #insert into dynamic memory and training
                    idx = self.find_insert_position()
                    if idx == -1: # no free memory position, replace an element already in memory
                        mingramloss = 1000
                        for j, mi in enumerate(self.memorylist):
                            if mi.pseudo_domain == domain:
                                loss = F.mse_loss(torch.tensor(item.current_grammatrix), torch.tensor(mi.current_grammatrix), reduction='mean')

                                if loss < mingramloss:
                                    mingramloss = loss
                                    idx = j
                        # print(self.memorylist[idx].scanner, 'replaced by', item.scanner, 'in domain', domain)
                    else:
                        self.domaincounter[domain] += 1
                    self.memorylist[idx] = item
                    self.labeling_counter += 1
                    
                    # print(item.img + epsilon)
                    # print(item.target + epsilon)
                    # print('model.get_task_metric(item.img + epsilon, item.target + epsilon)' + str(model.get_task_metric(item.img + epsilon, item.target + epsilon)))
                    
                    self.domainMetric[domain].append(model.get_task_metric(item.img + epsilon, item.target + epsilon))

                    # update center
                    domain_grams = [o.current_grammatrix for o in self.get_domainitems(domain)]

                    self.centers[domain] = np.array(domain_grams).mean(axis=0)
                    tmp_dist = [mean_squared_error(tr, self.centers[domain]) for tr in domain_grams]
                    self.max_center_distances[domain] = np.array(tmp_dist).mean() * 2

                    budget -= 1.0
                    # if budget % 5 == 0:
                    #     print('budgetI', budget)
                # else:
                #     if int(budget)<1:
                #         if not END:
                #             print('run out of budget ', budget)
                #             END = True

        return budget

    def check_pseudodomain(self, grammatrix):
        max_pred = 100000
        current_domain = -1

        for j, center in self.centers.items():
            if mean_squared_error(grammatrix, center)<self.max_center_distances[j]:
                current_pred = mean_squared_error(grammatrix, center)
                if current_pred<max_pred:
                    current_domain = j
                    max_pred = current_pred
        # print('predicted ', current_domain)
        return current_domain


    def get_domainitems(self, domain):
        items = []
        for mi in self.memorylist:
            if mi.pseudo_domain == domain:
                items.append(mi)
        return items


class UncertaintyDynamicMemory(DynamicMemory):

    def __init__(self, initelements, **kwargs):
        super(UncertaintyDynamicMemory, self).__init__(initelements, **kwargs)
        self.uncertainty_threshold = kwargs['uncertainty_threshold']
        self.forceitems = []

        if 'random_insert' in kwargs:
            self.random_insert = kwargs['random_insert']
        else:
            self.random_insert = False

        # print(self.random_insert, 'random insert')

    def insert_element(self, item, uncertainty, budget, model):
        if budget>0 and uncertainty>self.uncertainty_threshold:
            # print('insert element of type', item.scanner, budget)
            if len(self.memorylist)<self.memorymaximum:
                self.memorylist.append(item)
                self.forceitems.append(item)
                self.labeling_counter += 1
                budget-=1
            else:
                if self.random_insert:
                    insertidx = random.randint(0, len(self.memorylist)-1)

                    self.memorylist[insertidx] = item
                    self.forceitems.append(item)
                    self.labeling_counter += 1
                else:
                    assert (item.current_grammatrix is not None)
                    insertidx = -1
                    mingramloss = 1000
                    for j, mi in enumerate(self.memorylist):
                        loss = F.mse_loss(torch.tensor(item.current_grammatrix), torch.tensor(mi.current_grammatrix),
                                          reduction='mean')

                        if loss < mingramloss:
                            mingramloss = loss
                            insertidx = j
                    self.memorylist[insertidx] = item
                    self.forceitems.append(item)
                    self.labeling_counter += 1
                budget-=1

            scanners = dict()
            for mi in self.memorylist:
                if mi.scanner in scanners:
                    scanners[mi.scanner] += 1
                else:
                    scanners[mi.scanner] = 1
            # print(scanners)

        return budget

