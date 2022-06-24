
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import difflib
import json
import pickle as pkl
import csv
import numpy as np


# ----------------------------------------------------------------------------------------------------------------- #
class DogBreed(object):
    def __init__(self, abbrev, name_akc=None, name_stanext=None, name_xlsx=None, path_akc=None, path_stanext=None, ind_in_xlsx=None, ind_in_xlsx_matrix=None, ind_in_stanext=None, clade=None):
        self._abbrev = abbrev 
        self._name_xlsx = name_xlsx 
        self._name_akc = name_akc
        self._name_stanext = name_stanext
        self._path_stanext = path_stanext
        self._additional_names = set()
        if self._name_akc is not None:
            self.add_akc_info(name_akc, path_akc)
        if self._name_stanext is not None:
            self.add_stanext_info(name_stanext, path_stanext, ind_in_stanext)
        if self._name_xlsx is not None:
            self.add_xlsx_info(name_xlsx, ind_in_xlsx, ind_in_xlsx_matrix, clade)
    def add_xlsx_info(self, name_xlsx, ind_in_xlsx, ind_in_xlsx_matrix, clade):
        assert (name_xlsx is not None) and (ind_in_xlsx is not None) and (ind_in_xlsx_matrix is not None) and (clade is not None)
        self._name_xlsx = name_xlsx
        self._ind_in_xlsx = ind_in_xlsx
        self._ind_in_xlsx_matrix = ind_in_xlsx_matrix
        self._clade = clade
    def add_stanext_info(self, name_stanext, path_stanext, ind_in_stanext):
        assert (name_stanext is not None) and (path_stanext is not None) and (ind_in_stanext is not None)
        self._name_stanext = name_stanext
        self._path_stanext = path_stanext
        self._ind_in_stanext = ind_in_stanext
    def add_akc_info(self, name_akc, path_akc):
        assert (name_akc is not None) and (path_akc is not None)
        self._name_akc = name_akc
        self._path_akc = path_akc
    def add_additional_names(self, name_list):
        self._additional_names = self._additional_names.union(set(name_list)) 
    def add_text_info(self, text_height, text_weight, text_life_exp):
        self._text_height = text_height
        self._text_weight = text_weight
        self._text_life_exp = text_life_exp
    def get_datasets(self):
        # all datasets in which this breed is found
        datasets = set()
        if self._name_akc is not None:
            datasets.add('akc')
        if self._name_stanext is not None:
            datasets.add('stanext')
        if self._name_xlsx is not None:
            datasets.add('xlsx')
        return datasets
    def get_names(self):
        # set of names for this breed
        names = {self._abbrev, self._name_akc, self._name_stanext, self._name_xlsx, self._path_stanext}.union(self._additional_names)
        names.discard(None)
        return names
    def get_names_as_pointing_dict(self):
        # each name points to the abbreviation
        names = self.get_names()
        my_dict = {}
        for name in names:
            my_dict[name] = self._abbrev
        return my_dict
    def print_overview(self):
        # print important information to get an overview of the class instance
        if self._name_akc is not None:
            name = self._name_akc
        elif self._name_xlsx is not None:
            name = self._name_xlsx
        else:
            name = self._name_stanext
        print('----------------------------------------------------')
        print('----- dog breed: ' + name )
        print('----------------------------------------------------')
        print('[names]')
        print(self.get_names())
        print('[datasets]')
        print(self.get_datasets())
        # see https://stackoverflow.com/questions/9058305/getting-attributes-of-a-class
        print('[instance attributes]')
        for attribute, value in self.__dict__.items():
            print(attribute, '=', value)
    def use_dict_to_save_class_instance(self):
        my_dict = {}
        for attribute, value in self.__dict__.items():
            my_dict[attribute] = value
        return my_dict
    def use_dict_to_load_class_instance(self, my_dict):
        for attribute, value in my_dict.items():
            setattr(self, attribute, value)
        return 

# ----------------------------------------------------------------------------------------------------------------- #
def get_name_list_from_summary(summary):
    name_from_abbrev_dict = {}
    for breed in summary.values():
        abbrev = breed._abbrev
        all_names = breed.get_names()
        name_from_abbrev_dict[abbrev] = list(all_names)
    return name_from_abbrev_dict
def get_partial_summary(summary, part):
    assert part in ['xlsx', 'akc', 'stanext']
    partial_summary = {}
    for key, value in summary.items():
        if (part == 'xlsx' and value._name_xlsx is not None) \
            or (part == 'akc' and value._name_akc is not None) \
            or (part == 'stanext' and value._name_stanext is not None):
            partial_summary[key] = value
    return partial_summary
def get_akc_but_not_stanext_partial_summary(summary):
    partial_summary = {}
    for key, value in summary.items():
        if value._name_akc is not None:
            if value._name_stanext is None:
                partial_summary[key] = value
    return partial_summary    

# ----------------------------------------------------------------------------------------------------------------- #
def main_load_dog_breed_classes(path_complete_abbrev_dict_v1, path_complete_summary_breeds_v1):
    with open(path_complete_abbrev_dict_v1, 'rb') as file:
        complete_abbrev_dict = pkl.load(file)
    with open(path_complete_summary_breeds_v1, 'rb') as file: 
        complete_summary_breeds_attributes_only = pkl.load(file)
    
    complete_summary_breeds = {}
    for key, value in complete_summary_breeds_attributes_only.items():
        attributes_only = complete_summary_breeds_attributes_only[key]
        complete_summary_breeds[key] = DogBreed(abbrev=attributes_only['_abbrev'])
        complete_summary_breeds[key].use_dict_to_load_class_instance(attributes_only)
    return complete_abbrev_dict, complete_summary_breeds


# ----------------------------------------------------------------------------------------------------------------- #
def load_similarity_matrix_raw(xlsx_path):
    # --- LOAD EXCEL FILE FROM DOG BREED PAPER
    xlsx = pd.read_excel(xlsx_path)
    # create an array
    abbrev_indices = {}
    matrix_raw = np.zeros((168, 168))
    for ind in range(1, 169):
        abbrev = xlsx[xlsx.columns[2]][ind]
        abbrev_indices[abbrev] = ind-1
    for ind_col in range(0, 168):
        for ind_row in range(0, 168):
            matrix_raw[ind_col, ind_row] = float(xlsx[xlsx.columns[3+ind_col]][1+ind_row])
    return matrix_raw, abbrev_indices



# ----------------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------------------------------------- #
# load the (in advance created) final dict of dog breed classes
ROOT_PATH_BREED_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'data', 'breed_data')
path_complete_abbrev_dict_v1 = os.path.join(ROOT_PATH_BREED_DATA, 'complete_abbrev_dict_v2.pkl')
path_complete_summary_breeds_v1 = os.path.join(ROOT_PATH_BREED_DATA, 'complete_summary_breeds_v2.pkl')
COMPLETE_ABBREV_DICT, COMPLETE_SUMMARY_BREEDS = main_load_dog_breed_classes(path_complete_abbrev_dict_v1, path_complete_summary_breeds_v1)
# load similarity matrix, data from: 
#   Parker H. G., Dreger D. L., Rimbault M., Davis B. W., Mullen A. B., Carpintero-Ramirez G., and Ostrander E. A.
#   Genomic analyses reveal the influence of geographic origin, migration, and hybridization on modern dog breed 
#   development. Cell Reports, 4(19):697–708, 2017.
xlsx_path = os.path.join(ROOT_PATH_BREED_DATA, 'NIHMS866262-supplement-2.xlsx')
SIM_MATRIX_RAW, SIM_ABBREV_INDICES = load_similarity_matrix_raw(xlsx_path)

