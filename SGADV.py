# Targeted attack to face recognition system
import eagerpy as ep
import codecs
import time
import torch
from math import ceil
from torch import Tensor
from scipy.io import loadmat
from facenet_pytorch import InceptionResnetV1
from insightface import iresnet100
import foolbox.attacks as attacks
from foolbox.models import PyTorchModel
from foolbox.utils import samples, FMR, cos_similarity_score
import pytorch_ssim
import lpips
from torchvision.utils import save_image
from scipy.io import savemat

def main() -> None:
    # Settings
    samplesize = int(10)
    subject = int(2)
    foldersize = int(samplesize*subject/2)
    source = "lfw" # lfw, CelebA-HQ
    target = "CelebA-HQ" # lfw, CelebA-HQ
    dfr_model = 'facenet' # facenet, insightface
    threshold = 0.7032619898135847 # facenet: 0.7032619898135847; insightface: 0.5854403972629942
    attack_model = attacks.LinfPGD
    loss_type = 'ST' #'ST', 'C-BCE'
    epsilons = 0.03
    steps = 1000
    step_size = 0.001
    convergence_threshold = 0.0001
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    totalsize = samplesize*subject
    batchsize = 20
    
    # Log
    log_time = time.strftime("%Y%m%d%H%M%S",time.localtime())
    f = codecs.open(f'results/logs/{loss_type}_{source}_{target}_{dfr_model}_{attack_model.__module__}_{log_time}.txt','a','utf-8')
    f.write(f"samplesize = {samplesize}, subject = {subject}, source = {source}, target = {target}, dfr_model = {dfr_model}, threshold = {threshold}\n")
    f.write(f"attack_model = {attack_model}, loss_type = {loss_type}, epsilons = {epsilons}, steps = {steps}, step_size = {step_size}, convergence_threshold = {convergence_threshold}, batchsize = {batchsize}\n")
    
    # Model
    if dfr_model=='insightface':
        model = iresnet100(pretrained=True).eval()
    elif dfr_model=='facenet':
        model = InceptionResnetV1(pretrained='vggface2').eval()
    mean=[0.5]*3
    std=[0.5]*3
    preprocessing = dict(mean = mean, std = std, axis=-3)
    bounds=(0, 1)
    fmodel = PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

    # Load data
    features_tmp = loadmat(f'mat/{target}_{dfr_model}_templates.mat')[f'{target}_{dfr_model}_templates']
    features = Tensor(features_tmp)
    source_images, _ = ep.astensors(*samples(fmodel, dataset=f"{source}_test", batchsize=subject*samplesize, model_type=dfr_model))

    # Input data
    attack_index = list(range(samplesize*subject))
    attack_images = source_images[attack_index]
    target_index = list(range(foldersize,foldersize*2))+list(range(samplesize,foldersize))+list(range(0,samplesize))
    target_features = features[target_index]
    del source_images
    
    # Run attack
    attack = attack_model(threshold=threshold, loss_type=loss_type, steps=steps, abs_stepsize=step_size, convergence_threshold=convergence_threshold, device=device)
    raw_advs = Tensor([]).to(device)
    advs_features = Tensor([]).to(device)
    time_cost = 0
    for i in range(ceil(totalsize/batchsize)):
        print(f"Batch: {i+1}")
        start = i*batchsize
        if i == ceil(totalsize/batchsize)-1:
            batchsize = totalsize - batchsize*i
        start_time = time.time()
        raw_advs_tmp, _, _ = attack(fmodel, attack_images[start:start+batchsize], target_features[start:start+batchsize], epsilons=epsilons)
        end_time = time.time()
        time_cost = time_cost + end_time - start_time
        advs_features_tmp = fmodel(raw_advs_tmp)
        raw_advs = torch.cat((raw_advs, raw_advs_tmp.raw),0)
        advs_features = torch.cat((advs_features, advs_features_tmp.raw),0)
        del raw_advs_tmp, advs_features_tmp
    del attack, fmodel, model
    print(f"Attack costs {time_cost}s")
    f.write(f"Attack costs {time_cost}s\n")
    
    # Save advs template
    adv_template = advs_features.cpu().numpy()
    savemat(f'mat/{loss_type}_{source}_{target}_{dfr_model}_templates.mat', mdict={f"{loss_type}_{source}_{target}_{dfr_model}_templates": adv_template})
    
    # Save advs
    save_image(raw_advs[10], f'results/images/{loss_type}_{source}_{target}_{dfr_model}_{log_time}_adv.jpg')
    noise = (raw_advs[10]-attack_images[10].raw+bounds[1]-bounds[0])/((bounds[1]-bounds[0])*2)
    save_image(noise, f'results/images/{loss_type}_{source}_{target}_{dfr_model}_{log_time}_noise.jpg')
    del noise
    
    # Compute SSIM
    ssim_loss = pytorch_ssim.SSIM()
    ssim_score = ssim_loss(attack_images.raw.cpu(),raw_advs.cpu())
    print(f"SSIM = {ssim_score}")
    f.write(f"SSIM = {ssim_score}\n")
    del ssim_loss, ssim_score
    
    # Compute LPIPS
    loss_fn = lpips.LPIPS(net='alex')
    if bounds != (-1, 1):
        attack_images = attack_images.raw.cpu()*2-1
        raw_advs = raw_advs.cpu()*2-1
    lpips_score = loss_fn.forward(attack_images,raw_advs).mean()
    print(f"LPISP = {lpips_score}")
    f.write(f"LPISP = {lpips_score}\n")
    del attack_images, raw_advs, loss_fn
    
    #Compute dissimilarity
    dissimilarity = 1-cos_similarity_score(advs_features,target_features).mean()
    print(f"Dissimilarity = {dissimilarity}")
    f.write(f"Dissimilarity = {dissimilarity}\n")
    
    # Compute FMR
    fmr_target, fmr_renew = FMR(advs_features, target_features, threshold, samplesize)
    print("Attack performance:")
    f.write("Attack performance:\n")
    print(f" advs vs targets: FMR = {fmr_target * 100:.2f}%")
    f.write(f" advs vs targets: FAR = {fmr_target * 100:.2f}%\n")
    print(f" advs vs renews: FMR = {fmr_renew * 100:.2f}%")
    f.write(f" advs vs renews: FAR = {fmr_renew * 100:.2f}%\n")

    f.close()

if __name__ == "__main__":
    main()
