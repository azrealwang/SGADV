## This file must be modified to use!!!

from cv2 import imwrite, imread
import os
import numpy as np
from collections import Counter


def main() -> None:
    
    min_faces_per_person = 10 # only subjects with equal or more images will be selected, only the first {min_faces_per_person} will be selected
    subject = 158 # how many subjects to select
    
    data_home = r"C:\Users\{user_name}\Workspace\Git\{project_name}\Dataset" # replace with your own repository
    
    # The resources as belowed can be found from CelebA and CelebA-HQ
    txt_path = os.path.join(data_home, "CelebA-HQ")
    img_path = os.path.join(data_home, "CelebA-HQ\CelebA-HQ-img")
    output_path = os.path.join(data_home, "tmp\CelebA-HQ")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    celebA_file = np.loadtxt(f"{txt_path}\identity_CelebA.txt", dtype='str', skiprows=0, usecols=(0))
    celebA_id = np.loadtxt(f"{txt_path}\identity_CelebA.txt", dtype='int', skiprows=0, usecols=(1))
    celebA_HQ_file = np.loadtxt(f"{txt_path}\CelebA-HQ-to-CelebA-mapping.txt", dtype='str', skiprows=1, usecols=(2))
    celebA_HQ = np.loadtxt(f"{txt_path}\CelebA-HQ-to-CelebA-mapping.txt", dtype='str', skiprows=1, usecols=(0,1))

    print("Filtering subjects...")
    for m in range(len(celebA_HQ[:,0])):
        index = np.where(celebA_file == celebA_HQ_file[m])
        celebA_HQ[m,1] = celebA_id[index][0]
    id_list = celebA_HQ[:,1]
    count_dict = Counter(id_list)
    count_keys = np.array(list(count_dict.keys()))
    feltch_count_index = np.where(np.array(list(count_dict.values())) >= min_faces_per_person)[0]
    fetch_id = count_keys[feltch_count_index[subject:subject*2]]

    print("Generating images...")
    n=0
    cls = 0
    for i in range(len(fetch_id)):
        celebA_HQ_index = np.where(celebA_HQ[:,1]==fetch_id[i])[0]
        for j in range(min_faces_per_person):
            celebA_HQ_file_index = celebA_HQ_index[j]
            img = imread(f"{img_path}\{celebA_HQ_file_index}.jpg")
            imwrite(f'{output_path}\%05d_%d.jpg'%(n,cls),img)
            n=n+1
        cls = cls+1
    
if __name__ == "__main__":
    main()
