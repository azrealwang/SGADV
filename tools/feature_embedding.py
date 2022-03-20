from foolbox.utils import samples,cal_eer
from eagerpy import astensors
from scipy.io import savemat
from math import ceil
from facenet_pytorch import InceptionResnetV1
from insightface import iresnet100,iresnet50,iresnet34
from foolbox.models import PyTorchModel
from torch import Tensor
import codecs
import time


def main() -> None:    
    # Settings
    batchsize = 100
    samplesize = 10
    subject = 158
    totalsize = samplesize*subject
    dataset = "CelebA-HQ" # CelebA-HQ, lfw
    dfr_model = 'facenet' #facenet, insightface
    pre_model = 'vgg2face'
    shape = (160, 160)
    
    # Log
    log_time = time.strftime("%Y%m%d%H%M%S",time.localtime())
    f = codecs.open(f'results/logs/eer_{dataset}_{dfr_model}_{log_time}.txt','a','utf-8')
    f.write(f"samplesize = {samplesize}, subject = {subject}, dataset = {dataset}, dfr_model = {dfr_model}, shape = {shape}\n")
    
    # Model
    if dfr_model=='insightface':
        if pre_model=='50':
            model = iresnet50(pretrained=True).eval()
        elif pre_model=='34':
            model = iresnet34(pretrained=True).eval()
        else:
            model = iresnet100(pretrained=True).eval()
    elif dfr_model=='facenet':
        if pre_model=='casia':
            model = InceptionResnetV1(pretrained='casia-webface').eval()
        else:
            model = InceptionResnetV1(pretrained='vggface2').eval() 
    mean=[0.5]*3
    std=[0.5]*3
    preprocessing = dict(mean = mean, std = std, axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1),preprocessing=preprocessing)

    # Feature embedding
    features_tmp=list()
    labels_tmp=list()
    for i in range(ceil(totalsize/batchsize)):
        print(f"Batch: {i+1}")
        index = i*batchsize
        if i == ceil(totalsize/batchsize)-1:
            batchsize = totalsize - batchsize*i
        images_batch, labels_batch = astensors(*samples(fmodel, dataset=f"{dataset}_test", batchsize=batchsize, index=index, shape=shape))
        labels_tmp.extend(labels_batch.numpy())
        features_batch = fmodel(images_batch).numpy()  
        features_tmp.extend(features_batch)
    savemat(f'mat/{dataset}_{dfr_model}_templates.mat', mdict={f'{dataset}_{dfr_model}_templates': features_tmp})
    savemat(f'mat/{dataset}_{dfr_model}_labels.mat', mdict={f'{dataset}_{dfr_model}_labels': labels_tmp})
    
    # Compute EER and threshold
    eer, thresh = cal_eer(Tensor(features_tmp), Tensor(labels_tmp))
    print(f"EER = {eer}, threshold = {thresh}")
    f.write(f"EER = {eer}, threshold = {thresh}\n")
    
    f.close()
    
if __name__ == "__main__":
    main()
