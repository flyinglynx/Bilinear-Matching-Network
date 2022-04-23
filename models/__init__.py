import torch

from models.backbone import build_backbone
from models.counter import get_counter
from models.epf_extractor import build_epf_extractor
from models.refiner import build_refiner
from models.matcher import build_matcher
from models.class_agnostic_counting_model import CACModel

def build_model(cfg):
    backbone = build_backbone(cfg)
    epf_extractor = build_epf_extractor(cfg)
    refiner = build_refiner(cfg)
    matcher = build_matcher(cfg)
    counter = get_counter(cfg)
    model = CACModel(backbone, epf_extractor, refiner, matcher, counter, cfg.MODEL.hidden_dim)
    
    return model
    
    
    