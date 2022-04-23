from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Directory
# -----------------------------------------------------------------------------
_C.DIR = CN()
_C.DIR.dataset = "FSC147"
_C.DIR.exp = "dataset_model_inputblocksize_outputstride_resizeratio_cropsize_batchsize_epoch"
_C.DIR.snapshot = "./snapshots"
_C.DIR.result = "./results"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.name = "mtc"
_C.DATASET.list_train = "./data/train.txt"
_C.DATASET.list_val = "./data/val.txt"
_C.DATASET.list_test = "./data/test.txt"
# maximum input image size of long edge
_C.DATASET.max_size = 2048
# minimum input image size of long edge
_C.DATASET.min_size = 384 
# maxmimum downsampling rate of the network
_C.DATASET.padding_constant = 8
_C.DATASET.downsampling_rate = 1
# scale the ground truth density map
_C.DATASET.scaling = 1
# preload dataset into the memory to speed up training
_C.DATASET.preload = True
# randomly horizontally flip images when training
_C.DATASET.random_flip = True
# The size of resized exemplar patches 
_C.DATASET.exemplar_size = (128, 128)
_C.DATASET.exemplar_number = 3

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# architecture of net_encoder
_C.MODEL.backbone = "resnet50"
_C.MODEL.epf_extractor = "direct_pooling"
_C.MODEL.refiner = "self_similarity_module"
_C.MODEL.matcher = "dynamic_similairty_matcher"
_C.MODEL.counter = "local_count"
_C.MODEL.fix_bn = True
_C.MODEL.ep_scale_embedding = False
_C.MODEL.use_bias = False
_C.MODEL.ep_scale_number = 20
_C.MODEL.backbone_layer = "layer4"
_C.MODEL.hidden_dim = 256
_C.MODEL.refiner_proj_dim = 256
_C.MODEL.matcher_proj_dim = 256
_C.MODEL.dynamic_proj_dim = 128
_C.MODEL.dilation = False
_C.MODEL.refiner_layers = 6
_C.MODEL.matcher_layers = 6
_C.MODEL.repeat_times = 1
# dim of counter
_C.MODEL.counter_dim = 256
# use pretrained model
_C.MODEL.pretrain = True
# fix bn params, only under finetuning
_C.MODEL.fix_bn = False


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# restore training from a checkpoint
_C.TRAIN.resume = "model_ckpt.pth.tar"
# numbers of exemplar boxes
_C.TRAIN.exemplar_number = 3
# loss function
_C.TRAIN.counting_loss = "l1loss"
_C.TRAIN.contrast_loss = "info_nce"
# weight for contrast loss
_C.TRAIN.contrast_weight = 1e-5
# loss reduction
_C.TRAIN.loss_reduction = "mean"
# batch size
_C.TRAIN.batch_size = 1
# epochs to train for
_C.TRAIN.epochs = 20
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000
# optimizer and learning rate
_C.TRAIN.optimizer = "AdamW"
_C.TRAIN.lr_backbone = 0.01
_C.TRAIN.lr = 0.01
# milestone
_C.TRAIN.lr_drop = 200
# momentum
_C.TRAIN.momentum = 0.95
# weights regularizer
_C.TRAIN.weight_decay = 5e-4
# gradient clipping max norm
_C.TRAIN.clip_max_norm = 0.1
# number of data loading workers
_C.TRAIN.num_workers = 0
# frequency to display
_C.TRAIN.disp_iter = 20
# manual seed
_C.TRAIN.seed = 2020
_C.TRAIN.start_epoch = 0
_C.TRAIN.device = 'cuda:0'

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# the checkpoint to evaluate on
_C.VAL.resume = "model_best.pth.tar"
# currently only supports 1
_C.VAL.batch_size = 1
# frequency to display
_C.VAL.disp_iter = 10
# frequency to validate
_C.VAL.val_epoch = 10
# evaluate_only
_C.VAL.evaluate_only = False
_C.VAL.visualization = False