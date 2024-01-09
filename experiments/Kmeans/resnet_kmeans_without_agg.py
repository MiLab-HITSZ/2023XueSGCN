
import math
import torch
import torch.nn
import torchnet.meter as tnt

import copy
import os
import typing
from collections import OrderedDict
from federatedml.framework.homo.blocks import aggregator, random_padding_cipher
from federatedml.framework.homo.blocks.secure_aggregator import SecureAggregatorTransVar
from backend.gcn.models import *
from backend.multi_label.losses.AsymmetricLoss import *
from backend.utils.APMeter import AveragePrecisionMeter
from backend.utils.aggregators.aggregator import *
from backend.utils.loader.dataset_loader import DatasetLoader
from backend.utils.mylogger.mywriter import MyWriter
from federatedml.param.gcn_param import GCNParam
from federatedml.util import LOGGER
from federatedml.util.homo_label_encoder import HomoLabelEncoderArbiter

my_writer = MyWriter(dir_name=os.getcwd())

client_header = ['epoch', 'OP', 'OR', 'OF1', 'CP', 'CR', 'CF1', 'OP_3', 'OR_3', 'OF1_3', 'CP_3', 'CR_3', 'CF1_3', 'map',
                 'loss']
server_header = ['agg_iter', 'OP', 'OR', 'OF1', 'CP', 'CR', 'CF1', 'OP_3', 'OR_3', 'OF1_3', 'CP_3', 'CR_3', 'CF1_3',
                 'map', 'loss']
train_writer = my_writer.get("train.csv", header=client_header)
valid_writer = my_writer.get("valid.csv", header=client_header)
avgloss_writer = my_writer.get("avgloss.csv", header=server_header)

train_loss_writer = my_writer.get("train_loss.csv", header=['epoch', 'objective_loss', 'entropy_loss', 'overall_loss'])

scene_cnts_writer = my_writer.get("total_scene_cnts.csv")

centers_dir = os.path.join(os.path.join(os.getcwd(), 'stats'), 'centers')

if not os.path.exists(centers_dir):
    os.makedirs(centers_dir)


class _FedBaseContext(object):
    def __init__(self, max_num_aggregation, name):
        self._name = name

        
        
        self.max_num_aggregation = max_num_aggregation
        self._aggregation_iteration = 0

    
    
    def _suffix(self, group: str = "model"):
        return (
            self._name,
            group,
            f"{self._aggregation_iteration}",
        )

    def increase_aggregation_iteration(self):
        self._aggregation_iteration += 1

    @property
    def aggregation_iteration(self):
        return self._aggregation_iteration

    def finished(self):
        if self._aggregation_iteration >= self.max_num_aggregation:
            return True
        return False



class FedClientContext(_FedBaseContext):
    def __init__(self, max_num_aggregation, aggregate_every_n_epoch, name="default"):
        super(FedClientContext, self).__init__(max_num_aggregation=max_num_aggregation, name=name)
        self.name = name
        self.transfer_variable = SecureAggregatorTransVar()
        self.aggregator = aggregator.Client(self.transfer_variable.aggregator_trans_var)
        self.random_padding_cipher = random_padding_cipher.Client(
            self.transfer_variable.random_padding_cipher_trans_var
        )
        self.aggregate_every_n_epoch = aggregate_every_n_epoch
        self._params: list = []

        self._should_stop = False
        self.metrics_summary = []

    
    def init(self):
        self.name = self.random_padding_cipher.create_cipher()

    def encrypt(self, tensor: torch.Tensor, weight):
        return self.random_padding_cipher.encrypt(
            torch.clone(tensor).detach().mul_(weight)
        ).numpy()

    def tensor2arr(self, tensors):
        tensor_arrs = []
        for tensor in tensors:
            tensor_arr = tensor.data.cpu().numpy()
            tensor_arrs.append(tensor_arr)
        return tensor_arrs

    
    
    def send_model(self, tensors, bn_data, weight):
        tensor_arrs = self.tensor2arr(tensors)
        bn_arrs = self.tensor2arr(bn_data)
        self.aggregator.send_model(
            (tensor_arrs, bn_arrs, weight), suffix=self._suffix()
        )

    
    def recv_model(self):
        return self.aggregator.get_aggregated_model(suffix=self._suffix())

    
    def send_metrics(self, metrics, weight):
        self.aggregator.send_model((metrics, weight), suffix=self._suffix(group="metrics"))

    def recv_convergence(self):
        return self.aggregator.get_aggregated_model(
            suffix=self._suffix(group="convergence")
        )

    
    def do_aggregation(self, bn_data, weight, device):
        
        self.send_model(self._params, bn_data, weight)


        recv_elements: typing.List = self.recv_model()

        global_model, bn_data = recv_elements
        
        agg_tensors = []
        for arr in global_model:
            agg_tensors.append(torch.from_numpy(arr).to(device))
        for param, agg_tensor in zip(self._params, agg_tensors):
            
            if param.grad is None:
                continue
            param.data.copy_(agg_tensor)

        bn_tensors = []
        for arr in bn_data:
            bn_tensors.append(torch.from_numpy(arr).to(device))
        return bn_tensors

    
    def do_convergence_check(self, weight, metrics):
        self.send_metrics(metrics, weight)
        
        
        return False

    
    def configure_aggregation_params(self, optimizer):
        if optimizer is not None:
            self._params = [
                param
                
                for param_group in optimizer.param_groups  
                for param in param_group["params"]
            ]
            return

        raise TypeError(f"params and optimizer can't be both none")

    def should_aggregate_on_epoch(self, epoch_index):
        return (epoch_index + 1) % self.aggregate_every_n_epoch == 0

    def should_stop(self):
        return self._should_stop

    def set_converged(self):
        self._should_stop = True



class FedServerContext(_FedBaseContext):
    
    def __init__(self, max_num_aggregation, eps=0.0, name="default"):
        super(FedServerContext, self).__init__(
            max_num_aggregation=max_num_aggregation, name=name
        )
        self.transfer_variable = SecureAggregatorTransVar()
        self.aggregator = aggregator.Server(self.transfer_variable.aggregator_trans_var)
        self.random_padding_cipher = random_padding_cipher.Server(
            self.transfer_variable.random_padding_cipher_trans_var
        )
        self._eps = eps
        self._loss = math.inf

    def init(self, init_aggregation_iteration=0):
        self.random_padding_cipher.exchange_secret_keys()
        self._aggregation_iteration = init_aggregation_iteration

    
    def send_model(self, aggregated_arrs):
        self.aggregator.send_aggregated_model(aggregated_arrs, suffix=self._suffix())

    
    def recv_model(self):
        return self.aggregator.get_models(suffix=self._suffix())

    
    def send_convergence_status(self, mAP, status):
        self.aggregator.send_aggregated_model(
            (mAP, status), suffix=self._suffix(group="convergence")
        )

    def recv_metrics(self):
        return self.aggregator.get_models(suffix=self._suffix(group="metrics"))

    def do_convergence_check(self):
        loss_metrics_pairs = self.recv_metrics()
        total_metrics = None
        total_weight = 0.0

        for metrics, weight in loss_metrics_pairs:
            cur_metrics = [metric * weight for metric in metrics]
            if total_metrics is None:
                total_metrics = cur_metrics
            else:
                total_metrics = [x + y for x, y in zip(total_metrics, cur_metrics)]  
            total_weight += weight

        
        mean_metrics = [metric / total_weight for metric in total_metrics]

        avgloss_writer.writerow([self.aggregation_iteration] + mean_metrics)

        mean_loss = mean_metrics[-1]

        is_converged = abs(mean_loss - self._loss) < self._eps

        self._loss = mean_metrics[-1]
        LOGGER.info(f"convergence check: loss={mean_loss}, is_converged={is_converged}")
        return is_converged, mean_loss


def build_aggregator(param: GCNParam, init_iteration=0):
    context = FedServerContext(
        max_num_aggregation=param.max_iter,
        eps=param.early_stop_eps
    )
    context.init(init_aggregation_iteration=init_iteration)
    fed_aggregator = GCNFedAggregator(context)
    return fed_aggregator


def build_fitter(param: GCNParam, train_data, valid_data):
    
    param.batch_size = 2
    param.max_iter = 1000
    param.num_labels = 80
    param.device = 'cuda:0'
    param.lr = 0.0001
    category_dir = '/home/klaus125/research/fate/my_practice/dataset/coco'

    epochs = param.aggregate_every_n_epoch * param.max_iter
    context = FedClientContext(
        max_num_aggregation=param.max_iter,
        aggregate_every_n_epoch=param.aggregate_every_n_epoch
    )
    
    context.init()
    inp_name = 'coco_glove_word2vec.pkl'
    

    batch_size = param.batch_size
    dataset_loader = DatasetLoader(category_dir, train_data.path, valid_data.path, inp_name=inp_name)

    
    train_loader, valid_loader = dataset_loader.get_loaders(batch_size)

    fitter = GCNFitter(param, epochs, context=context)
    return fitter, train_loader, valid_loader


class GCNFedAggregator(object):
    def __init__(self, context: FedServerContext):
        self.context = context
        self.model = None
        self.bn_data = None

    def fit(self, loss_callback):
        while not self.context.finished():
            recv_elements: typing.List[typing.Tuple] = self.context.recv_model()
            cur_iteration = self.context.aggregation_iteration

            tensors = [party_tuple[0] for party_tuple in recv_elements]
            bn_tensors = [party_tuple[1] for party_tuple in recv_elements]

            degrees = [party_tuple[2] for party_tuple in recv_elements]


            self.bn_data = aggregate_bn_data(bn_tensors, degrees)

            self.model = aggregate_whole_model(tensors, degrees)

            self.context.send_model((self.model, self.bn_data))
            self.context.do_convergence_check()

            self.context.increase_aggregation_iteration()

        if self.context.finished():
            print(os.getcwd())
            np.save('global_model', self.model)
            np.save('bn_data', self.bn_data)

    def export_model(self, param):
        pass

    @classmethod
    def load_model(cls, model_obj, meta_obj, param):
        param.restore_from_pb(meta_obj.params)

    pass

    @classmethod
    def load_model(cls, model_obj, meta_obj, param):
        pass

    @staticmethod
    def dataset_align():
        LOGGER.info("start label alignment")
        label_mapping = HomoLabelEncoderArbiter().label_alignment()
        LOGGER.info(f"label aligned, mapping: {label_mapping}")



class GCNFitter(object):
    def __init__(
            self,
            param,
            epochs,
            label_mapping=None,
            context: FedClientContext = None
    ):
        self.scheduler = ...
        self.param = copy.deepcopy(param)
        self._all_consumed_data_aggregated = True
        self.context = context
        self.label_mapping = label_mapping

        
        self.model, self.scheduler, self.optimizer, self.gcn_optimizer = _init_gcn_learner(self.param,
                                                                                           self.param.device)

        
        self.criterion = AsymmetricLossOptimized().to(self.param.device)

        self.start_epoch, self.end_epoch = 0, epochs

        
        
        self._num_data_consumed = 0
        
        
        self._num_label_consumed = 0
        
        self._num_per_labels = [0] * self.param.num_labels

        
        self.ap_meter = AveragePrecisionMeter(difficult_examples=False)

        self.lr_scheduler = None
        self.gcn_lr_scheduler = None

    def get_label_mapping(self):
        return self.label_mapping

    
    def fit(self, train_loader, valid_loader):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.on_fit_epoch_start(epoch, len(train_loader.sampler))
            valid_metrics = self.train_validate(epoch, train_loader, valid_loader, self.scheduler)
            self.on_fit_epoch_end(epoch, valid_loader, valid_metrics)
            if self.context.should_stop():
                break

    def on_fit_epoch_start(self, epoch, num_samples):
        if self._all_consumed_data_aggregated:
            self._num_data_consumed = num_samples
            self._all_consumed_data_aggregated = False
        else:
            self._num_data_consumed += num_samples

    def on_fit_epoch_end(self, epoch, valid_loader, valid_metrics):
        metrics = valid_metrics
        if self.context.should_aggregate_on_epoch(epoch):
            self.aggregate_model(epoch)
            status = self.context.do_convergence_check(
                len(valid_loader.sampler), metrics
            )
            if status:
                self.context.set_converged()
            self._all_consumed_data_aggregated = True

            self._num_data_consumed = 0
            self._num_label_consumed = 0
            self._num_per_labels = [0] * self.param.num_labels

            self.context.increase_aggregation_iteration()

    
    def train_one_epoch(self, epoch, train_loader, scheduler):
        
        self.ap_meter.reset()
        
        metrics = self.train(train_loader, self.model, self.criterion, self.optimizer, epoch, self.param.device,
                             scheduler)
        return metrics

    def validate_one_epoch(self, epoch, valid_loader, scheduler):
        self.ap_meter.reset()
        metrics = self.validate(valid_loader, self.model, self.criterion, epoch, self.param.device, scheduler)
        return metrics

    def aggregate_model(self, epoch):
        
        self.context.configure_aggregation_params(self.optimizer)
        
        

        
        bn_data = []
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                bn_data.append(layer.running_mean)
                bn_data.append(layer.running_var)

        
        weight_list = list(self._num_per_labels)
        weight_list.append(self._num_data_consumed)

        
        scene_cnts_writer.writerow([epoch] + self.model.total_scene_cnts)

        
        agg_bn_data = self.context.do_aggregation(weight=weight_list, bn_data=bn_data,
                                                              device=self.param.device)
        idx = 0
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                layer.running_mean.data.copy_(agg_bn_data[idx])
                idx += 1
                layer.running_var.data.copy_(agg_bn_data[idx])
                idx += 1

    def train_validate(self, epoch, train_loader, valid_loader, scheduler):
        self.train_one_epoch(epoch, train_loader, scheduler)
        valid_metrics = None
        if valid_loader:
            valid_metrics = self.validate_one_epoch(epoch, valid_loader, scheduler)
        if self.scheduler:
            self.scheduler.on_epoch_end(epoch, self.optimizer)
        return valid_metrics

    def train(self, train_loader, model, criterion, optimizer, epoch, device, scheduler):

        total_samples = len(train_loader.sampler)
        batch_size = 1 if total_samples < train_loader.batch_size else train_loader.batch_size
        steps_per_epoch = math.ceil(total_samples / batch_size)

        model.train()
        
        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        ENTROPY_LOSS_KEY = 'Entropy Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter()),
                              (ENTROPY_LOSS_KEY, tnt.AverageValueMeter())])

        sigmoid_func = torch.nn.Sigmoid()  
        for train_step, ((features, inp), target) in enumerate(train_loader):
            
            features = features.to(device)
            target = target.to(device)
            inp = inp.to(device)

            self._num_per_labels += target.t().sum(dim=1).cpu().numpy()

            
            self._num_label_consumed += target.sum().item()

            
            output = model(features, inp, y=target)

            predicts = output['output']
            
            self.ap_meter.add(predicts.data, target)

            objective_loss = criterion(sigmoid_func(predicts), target)

            overall_loss = objective_loss

            losses[OVERALL_LOSS_KEY].add(overall_loss.item())
            losses[OBJECTIVE_LOSS_KEY].add(objective_loss.item())
            optimizer.zero_grad()

            overall_loss.backward()

            optimizer.step()

            

            
            

        
        if (epoch + 1) % 4 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9

        mAP, _ = self.ap_meter.value()
        mAP *= 100
        loss = losses[OBJECTIVE_LOSS_KEY].mean
        

        
        
        torch.save(self.model.centers, os.path.join(centers_dir, f'centers_{epoch}.pth'))

        OP, OR, OF1, CP, CR, CF1 = self.ap_meter.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.ap_meter.overall_topk(3)
        metrics = [OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, mAP.item(), loss]
        train_writer.writerow([epoch] + metrics)

        train_loss_writer.writerow(
            [epoch, losses[OBJECTIVE_LOSS_KEY].mean, losses[ENTROPY_LOSS_KEY].mean, losses[OVERALL_LOSS_KEY].mean])
        return metrics

    def validate(self, valid_loader, model, criterion, epoch, device, scheduler):
        total_samples = len(valid_loader.sampler)
        batch_size = valid_loader.batch_size
        steps = math.ceil(total_samples / batch_size)

        OVERALL_LOSS_KEY = 'Overall Loss'
        OBJECTIVE_LOSS_KEY = 'Objective Loss'
        losses = OrderedDict([(OVERALL_LOSS_KEY, tnt.AverageValueMeter()),
                              (OBJECTIVE_LOSS_KEY, tnt.AverageValueMeter())])
        sigmoid_func = torch.nn.Sigmoid()
        model.eval()
        self.ap_meter.reset()

        with torch.no_grad():
            for validate_step, ((features, inp), target) in enumerate(valid_loader):
                features = features.to(device)
                inp = inp.to(device)
                target = target.to(device)

                output = model(features, inp, y=target)
                predicts = output['output']

                
                self.ap_meter.add(predicts.data, target)

                objective_loss = criterion(sigmoid_func(predicts), target)

                losses[OBJECTIVE_LOSS_KEY].add(objective_loss.item())
                
                

        mAP, _ = self.ap_meter.value()
        mAP *= 100
        loss = losses[OBJECTIVE_LOSS_KEY].mean
        OP, OR, OF1, CP, CR, CF1 = self.ap_meter.overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.ap_meter.overall_topk(3)
        metrics = [OP, OR, OF1, CP, CR, CF1, OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k, mAP.item(), loss]
        valid_writer.writerow([epoch] + metrics)
        return metrics


def _init_gcn_learner(param, device='cpu'):
    
    
    
    num_scenes = param.num_scenes  
    
    lr, lrp = param.lr, 0.1

    model = resnet_kmeans(param.pretrained, device, num_scenes=num_scenes)
    gcn_optimizer = None

    
    optimizer = torch.optim.AdamW(model.get_config_optim(lr=lr, lrp=lrp), lr=param.lr, weight_decay=1e-4)

    scheduler = None
    return model, scheduler, optimizer, gcn_optimizer
