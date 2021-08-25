from typing import Optional, Tuple, Any
import eagerpy as ep
import warnings
import os
import numpy as np
import time
import math
import torch
from torch.nn import CosineSimilarity
from torch import Tensor
from foolbox.plot import images
from matplotlib.pyplot import savefig
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from .types import Bounds
from .models import Model

def accuracy(fmodel: Model, inputs: Any, labels: Any) -> float:
    inputs_, labels_ = ep.astensors(inputs, labels)
    del inputs, labels

    predictions = fmodel(inputs_).argmax(axis=-1)
    accuracy = (predictions == labels_).float32().mean()
    return accuracy.item()

def samples(
    fmodel: Model,
    dataset: str = "lfw",
    index: int = 0,
    batchsize: int = 1,
    data_format: Optional[str] = None,
    bounds: Optional[Bounds] = None,
    shape: Optional[Tuple[int, int]] = None,
    model_type: Optional[str] = "facenet",
) -> Any:
    if hasattr(fmodel, "data_format"):
        if data_format is None:
            data_format = fmodel.data_format  # type: ignore
        elif data_format != fmodel.data_format:  # type: ignore
            raise ValueError(
                f"data_format ({data_format}) does not match model.data_format ({fmodel.data_format})"  # type: ignore
            )
    elif data_format is None:
        raise ValueError(
            "data_format could not be inferred, please specify it explicitly"
        )

    if bounds is None:
        bounds = fmodel.bounds

    images, labels = _samples(
        dataset=dataset,
        index=index,
        batchsize=batchsize,
        data_format=data_format,
        bounds=bounds,
        shape=shape,
        model_type=model_type,
    )
    
    if hasattr(fmodel, "dummy") and fmodel.dummy is not None:  # type: ignore
        images = ep.from_numpy(fmodel.dummy, images).raw  # type: ignore
        labels = ep.from_numpy(fmodel.dummy, labels).raw  # type: ignore
    else:
        warnings.warn(f"unknown model type {type(fmodel)}, returning NumPy arrays")
    return images, labels


def _samples(
    dataset: str,
    index: int,
    batchsize: int,
    data_format: str,
    bounds: Bounds,
    shape: Tuple[int, int],
    model_type: str
) -> Tuple[Any, Any]:   

    from PIL import Image

    images, labels = [], []
    basepath = r"data"
    samplepath = os.path.join(basepath, f"{dataset}")
    files = os.listdir(samplepath)
    for idx in range(index, index + batchsize):
        i = idx

        # get filename and label
        file = [n for n in files if f"{i:05d}_" in n][0]
        label = int(file.split(".")[0].split("_")[-1])

        # open file
        path = os.path.join(samplepath, file)
        image = Image.open(path)
        
        if model_type == "insightface":
            image = image.resize((112, 112))
        elif shape is not None:
            image = image.resize(shape)
        
        image = np.asarray(image, dtype=np.float32)

        if image.ndim == 2:
            image = image[..., np.newaxis]

        assert image.ndim == 3

        if data_format == "channels_first":
            image = np.transpose(image, (2, 0, 1))

        images.append(image)
        labels.append(label)

    images_ = np.stack(images)
    labels_ = np.array(labels)

    if bounds != (0, 255):
        images_ = images_ / 255 * (bounds[1] - bounds[0]) + bounds[0]
    return images_, labels_

def cos_similarity_score(featuresA: Tensor, featuresB: Tensor) -> Tensor:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cos = CosineSimilarity(dim=1,eps=1e-6)
    featuresA = featuresA.to(device)
    featuresB = featuresB.to(device)
    similarity = (cos(featuresA,featuresB)+1)/2
    del featuresA, featuresB
    return similarity

def cal_eer(features: Tensor, labels: Tensor) -> Tuple[Any, Any]:
    geniue_index1 = list()
    geniue_index2 = list()
    imposter_index1 = list()
    imposter_index2 = list()
    print("    Generating geniue/imposter labels...")
    for i in range(labels.shape[0]):
        for j in range(i+1,labels.shape[0]):
            if labels[i]==labels[j]:
                geniue_index1.extend([i])
                geniue_index2.extend([j])
            else:
                imposter_index1.extend([i])
                imposter_index2.extend([j])
    if len(geniue_index1) ==0 or len(imposter_index1)==0:
        raise RuntimeError("single class or single sample dataset")
    print("    Computing similarity score...")
    geniue_score = cos_similarity_score(features[geniue_index1],features[geniue_index2]).cpu()
    imposter_score = cos_similarity_score(features[imposter_index1],features[imposter_index2]).cpu()
    print("    Computing EER and Threshold...")
    fpr, tpr, thresholds = roc_curve([1]*len(geniue_score)+[0]*len(imposter_score), torch.cat((geniue_score,imposter_score),0), pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    
    return eer, thresh

def false_rate(
    featuresA: Tensor, 
    labelsA: Tensor, 
    featuresB: Tensor, 
    lablesB: Tensor,
    thresh: float,
) -> Tuple[Any, Any]:
    geniue_indexA = list()
    geniue_indexB = list()
    imposter_indexA = list()
    imposter_indexB = list()
    print("    Generating geniue/imposter labels...")
    for i in range(labelsA.shape[0]):
        for j in range(lablesB.shape[0]):
            if labelsA[i]==lablesB[j]:
                geniue_indexA.extend([i])
                geniue_indexB.extend([j])
            else:
                imposter_indexA.extend([i])
                imposter_indexB.extend([j])
    print("    Computing similarity score...")
    geniue_score = cos_similarity_score(featuresA[geniue_indexA],featuresB[geniue_indexB])
    imposter_score = cos_similarity_score(featuresA[imposter_indexA],featuresB[imposter_indexB])
    print("    Computing FAR and FRR...")
    frr = -1
    if len(geniue_score)>0:
        frr = np.float32(geniue_score <= thresh).mean()
    far = -1
    if len(imposter_score)>0:
        far = np.float32(imposter_score >= thresh).mean()
    return far, frr
    
    
def save_image(
    features: Any, 
    prefix: str,
    nrows: int = 1,
    log_time: Optional[str] = None
) -> None:
    if log_time is None:
        log_time = time.strftime("%Y%m%d%H%M%S",time.localtime())
    images(features,nrows=nrows)
    savefig(f'results/images/{prefix}_{log_time}.jpg')
    
    
def FMR(
    advs: Tensor, 
    targets: Tensor, 
    thresh: float,
    samplesize: int,
) -> Tuple[Any, Any]:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("    Generating score: advs vd target...")
    imposter_score_target = cos_similarity_score(advs,targets)
    
    print("    Generating score: advs vd renew...")
    imposter_score_renew = Tensor([]).to(device)
    totalsize = np.int(targets.shape[0])
    for i in range(samplesize):
        index_i = list(range(0+i,totalsize,samplesize))
        for j in range(samplesize): 
            if i!=j:
                index_j = list(range(0+j,totalsize,samplesize))
                similarity_tmp = cos_similarity_score(advs[index_i],targets[index_j])
                imposter_score_renew = torch.cat((imposter_score_renew,similarity_tmp),0)
    
    print("    Computing FMR...")
    fmr_target = -1
    if len(imposter_score_target)>0:
        fmr_target = np.float32(imposter_score_target.cpu() >= thresh).mean()
    fmr_renew = -1
    if len(imposter_score_renew)>0:
        fmr_renew = np.float32(imposter_score_renew.cpu() >= thresh).mean()
    del imposter_score_target, imposter_score_renew
    
    return fmr_target, fmr_renew


def convergence(loss_queue,convergence_threshold) -> bool:
    length = len(loss_queue)
    if length<loss_queue.maxlen:
        decision = False
    else:
        loss_queue = Tensor(loss_queue)
        delta_queue = loss_queue[1:length]-loss_queue[0:length-1]
        half_size = math.floor((length-1)/2)
        if abs(loss_queue[length-1])==0:
            decision = True
        elif all(i < convergence_threshold for i in abs(delta_queue)):
            decision = True
        elif sum(np.float32(delta_queue<=0))==half_size:
            decision = True
        else:
            decision = False
    
    return decision
    