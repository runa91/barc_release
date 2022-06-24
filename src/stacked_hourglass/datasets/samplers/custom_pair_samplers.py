
import numpy as np
import random
import copy
import time
import warnings

from torch.utils.data import Sampler
from torch._six import int_classes as _int_classes

class CustomPairBatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of indices.
    The structure of this sampler is way to complicated because it is a shorter/simplified version of 
    CustomBatchSampler. The relations between breeds are not relevant for the cvpr 2022 paper, but we kept 
    this structure which we were using for the experiments with clade related losses. ToDo: restructure 
    this sampler. 
    Args:
        data_sampler_info (dict): a dictionnary, containing information about the dataset and breeds. 
        batch_size (int): Size of mini-batch.
    """

    def __init__(self, data_sampler_info, batch_size):
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        assert batch_size%2 == 0
        self.data_sampler_info = data_sampler_info
        self.batch_size = batch_size
        self.n_desired_batches = int(np.floor(len(self.data_sampler_info['name_list']) / batch_size))        # 157

    def get_description(self):
        description = "\
            This sampler works only for even batch sizes. \n\
            It returns pairs of dogs of the same breed"
        return description


    def __iter__(self):
        breeds_summary = self.data_sampler_info['breeds_summary']

        breed_image_dict_orig = {}
        for img_name in self.data_sampler_info['name_list']:     # ['n02093859-Kerry_blue_terrier/n02093859_913.jpg', ... ]
            folder_name = img_name.split('/')[0]
            breed_name = folder_name.split(folder_name.split('-')[0] + '-')[1]
            if not (breed_name in breed_image_dict_orig):
                breed_image_dict_orig[breed_name] = [img_name]
            else:
                breed_image_dict_orig[breed_name].append(img_name)

        lengths = np.zeros((len(breed_image_dict_orig.values())))     
        for ind, value in enumerate(breed_image_dict_orig.values()): 
            lengths[ind] = len(value)

        sim_matrix_raw = self.data_sampler_info['breeds_sim_martix_raw']
        sim_matrix_raw[sim_matrix_raw>0].shape      # we have 1061 connections

        # from ind_in_sim_mat to breed_name
        inverse_sim_dict = {}
        for abbrev, ind in self.data_sampler_info['breeds_sim_abbrev_inds'].items():
            # breed_name might be None
            breed = breeds_summary[abbrev]
            breed_name = breed._name_stanext
            inverse_sim_dict[ind] = {'abbrev': abbrev,
                                    'breed_name': breed_name}

        # similarity for relevant breeds only:
        related_breeds_top_orig = {}
        temp = np.arange(sim_matrix_raw.shape[0])
        for breed_name, breed_images in breed_image_dict_orig.items():
            abbrev = self.data_sampler_info['breeds_abbrev_dict'][breed_name]
            related_breeds = {}
            if abbrev in self.data_sampler_info['breeds_sim_abbrev_inds'].keys():
                ind_in_sim_mat = self.data_sampler_info['breeds_sim_abbrev_inds'][abbrev]
                row = sim_matrix_raw[ind_in_sim_mat, :]
                rel_inds = temp[row>0]
                for ind in rel_inds:
                    rel_breed_name = inverse_sim_dict[ind]['breed_name']
                    rel_abbrev = inverse_sim_dict[ind]['abbrev'] 
                    # does this breed exist in this dataset?
                    if (rel_breed_name is not None) and (rel_breed_name in breed_image_dict_orig.keys()) and not (rel_breed_name==breed_name):
                        related_breeds[rel_breed_name] = row[ind]
            related_breeds_top_orig[breed_name] = related_breeds

        breed_image_dict = copy.deepcopy(breed_image_dict_orig)
        related_breeds_top = copy.deepcopy(related_breeds_top_orig)

        # clean the related_breeds_top dict such that it only contains breeds which are available
        for breed_name, breed_images in breed_image_dict.items():
            if len(breed_image_dict[breed_name]) < 1:
                for breed_name_rel in list(related_breeds_top[breed_name].keys()):
                    related_breeds_top[breed_name_rel].pop(breed_name, None)
                    related_breeds_top[breed_name].pop(breed_name_rel, None)
            
        # 1) build pairs of dogs
        set_of_breeds_with_at_least_2 = set() 
        for breed_name, breed_images in breed_image_dict.items():
            if len(breed_images) >= 2:
                set_of_breeds_with_at_least_2.add(breed_name)

        n_unused_images = len(self.data_sampler_info['name_list'])
        all_dog_duos = []
        n_new_duos = 1
        while n_new_duos > 0:
            for breed_name, breed_images in breed_image_dict.items():
                # shuffle image list for this specific breed (this changes the dict)
                random.shuffle(breed_images)
            breed_list = list(related_breeds_top.keys())
            random.shuffle(breed_list)
            n_new_duos = 0
            for breed_name in breed_list:
                if len(breed_image_dict[breed_name]) >= 2:
                    dog_a = breed_image_dict[breed_name].pop()
                    dog_b = breed_image_dict[breed_name].pop()
                    dog_duo = [dog_a, dog_b]
                    all_dog_duos.append({'image_names': dog_duo})      
                    # clean the related_breeds_top dict such that it only contains breeds which are still available
                    if len(breed_image_dict[breed_name]) < 1:
                        for breed_name_rel in list(related_breeds_top[breed_name].keys()):
                            related_breeds_top[breed_name_rel].pop(breed_name, None)
                            related_breeds_top[breed_name].pop(breed_name_rel, None)
                    n_new_duos += 1
                    n_unused_images -= 2

        image_name_to_ind = {}
        for ind_img_name, img_name in enumerate(self.data_sampler_info['name_list']):     
            image_name_to_ind[img_name] = ind_img_name

        # take all images and create the batches
        n_avail_2 = len(all_dog_duos)
        all_batches = []
        ind_in_duos = 0
        n_imgs_used_twice = 0
        for ind_b in range(0, self.n_desired_batches):
            batch_with_image_names = []
            for ind in range(int(np.floor(self.batch_size / 2))):
                if ind_in_duos >= n_avail_2:
                    ind_rand = random.randint(0, n_avail_2-1)
                    batch_with_image_names.extend(all_dog_duos[ind_rand]['image_names'])
                    n_imgs_used_twice += 2
                else:
                    batch_with_image_names.extend(all_dog_duos[ind_in_duos]['image_names'])
                ind_in_duos += 1


            batch_with_inds = []
            for image_name in batch_with_image_names:   # rather a folder than name
                batch_with_inds.append(image_name_to_ind[image_name])

            all_batches.append(batch_with_inds)

        for batch in all_batches:
            yield batch

    def __len__(self):
        # Since we are sampling pairs of dogs and not each breed has an even number of dogs, we can not 
        # guarantee to show each dog exacly once. What we do instead, is returning the same amount of 
        # batches as we would return with a standard sampler which is not based on dog pairs.    
        '''if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore'''
        return self.n_desired_batches








