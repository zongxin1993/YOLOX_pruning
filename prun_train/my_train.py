import time
import torch

from yolox.core.trainer import Trainer
import torch.nn as nn

from yolox.models.network_blocks import Bottleneck

class MY_Trainer_Loose(Trainer):
    def __init__(self, exp, args):
        super().__init__(exp,args)
    

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

    
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        ignore_bn_list = []
        for name, m in self.model.named_modules():
            if isinstance(m, Bottleneck):
                if m.use_add:
                    ignore_bn_list.append(name.rsplit(".", 2)[0]+".conv1.bn")
                    ignore_bn_list.append(name.rsplit(".", 2)[0]+".conv2.bn")
                    ignore_bn_list.append(name.rsplit(".", 2)[0]+".conv3.bn")
                    ignore_bn_list.append(name + '.conv1.bn')
                    ignore_bn_list.append(name + '.conv2.bn')
            if isinstance(m, nn.BatchNorm2d) and (name not in ignore_bn_list):
                m.weight.grad.data.add_(0.0001 * torch.sign(m.weight.data))  # L1
                m.bias.grad.data.add_(0.0001*10 * torch.sign(m.bias.data))  # L1
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

class MY_Trainer_Fine(Trainer):
    def __init__(self, exp, args):
        super().__init__(exp,args)
    
    def train(self,model):
        self.before_train(model)
        try:
            self.train_in_epoch()
        except Exception as e:
            raise
        finally:
            self.after_train()
    
    def before_train(self,model=None):
        from loguru import logger
        from yolox.utils import (
            ModelEMA,
            get_model_info,
            occupy_mem,
        )
        from yolox.data import DataPrefetcher
        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.utils.tensorboard import SummaryWriter

        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        if model is None:
            model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        logger.info("\n{}".format(model))

