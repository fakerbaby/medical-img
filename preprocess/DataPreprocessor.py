import os
import shutil
import re
from fnmatch import fnmatch
import logging
from pathlib import Path
import csv
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataPreprocessor():
    def __init__(self, path):
        self.path = path
        if os.path.isdir(path):
            self.datalist = os.listdir(path)
        else:
            try:
                realpath = os.path.dirname(__file__)
                path = os.path.join(realpath, path)
                os.mkdir(path=path)
            except OSError as e:
                print(e.strerror)
        self.dict = []

    
    
    def check_for_unification_and_repair(self):
        pass


    def uncompress_files(self):
        if "data" not in os.listdir(self.path):
            try:
                os.mkdir("dataset")
            except OSError as e:
                print("\"data\"", e.strerror)
         
        path = os.path.join(self.path, 'dataset')
        if os.path.isdir(path):
            dicts = os.listdir(path)
            for dict in dicts:
                if fnmatch(dict, "*.zip"):
                    os.system(f"unzip -O CP936 \'{path}/*.zip\' -d {Path(path)}")
                    break
        print("uncompress complete!")

    def data_denoising(self):
        pass

    def all_merge_walk(self, name):
        """
        os.walk tranversal all files in nested dirs 
        """
        base_path = os.path.join(self.path, name)
        for root, dirs, files in os.walk(base_path):
            for name in files:
                if fnmatch(name, "*.bmp") or fnmatch(name, "*.jpg") or fnmatch(name, "*.dcm") or  fnmatch(name, "*.tiff") or fnmatch(name, "[A-Z]*") and not fnmatch(name, "*.docx"):
                    print(name)
                    self.dict.append(os.path.join(root, name))

    def all_test_merge_walk(self, name):
        base_path = os.path.join(self.path, name)
        for root, dirs, files in os.walk(base_path):
            for name in files:
                if fnmatch(name, "*.bmp") or fnmatch(name, "*.jpg") or fnmatch(name, "*.dcm") or fnmatch(name, "*.tiff") or fnmatch(name, "[A-Z]*") and not fnmatch(name, "*.docx"):
                    if find_by_pattern(name, "N") and find_by_pattern(name, "T") is not True:
                        print(name)
                        self.dict.append(os.path.join(root, name))

  

    def collect_datalist(self, name):
        """
        collect data by self.dict(datapath list)
        """
        target_path = os.path.join(self.path, name)

        try:
            os.mkdir(target_path)
        except OSError as e:    
            print(e.strerror)
        for sub_path in self.dict:
            try:
                shutil.move(sub_path, target_path)
            except OSError as e:
                print(f'{sub_path}', e.strerror)


    def get_origin_datalist(self):
        if len(self.dict) == 0:
            return None
        return self.dict


    def generate_csv(self, target="origin", name='label'):
        # path = os.path.join(self.path, 'dataset')
        names = ['img', 'label']
        label = []
        csv_path = os.path.join(self.path, 'csv')
        target_path = os.path.join(self.path, target)
        try:
            os.mkdir(csv_path)
        except FileExistsError as e:
            print(e.strerror)
        bmp_list = os.listdir(path=target_path)
        for bmp in bmp_list:
            if find_by_pattern(bmp, pattern="train"):
                label.append(0)
            else:
                label.append(1)
        label_dict = dict(zip(bmp_list, label)) 
        print(label_dict)
        csv_file_path = os.path.join(csv_path, f'{name}.csv')
        try: 
            with open(csv_file_path, 'w') as f:
                w = csv.writer(f)
                for k,v in label_dict.items():
                    w.writerow([k,v])
        except OSError as e:
            print(e.strerror)
        #add_header
        df = pd.read_csv(csv_file_path, names=names)
        df.to_csv(csv_file_path, index=False)
        print("csv generation complete!")

    def generate_test_csv(self, target="test_n", name='label_n'):
        names = ['img', 'label']
        label = []
        csv_path = os.path.join(self.path, 'csv')
        target_path = os.path.join(self.path, target)
        try:
            os.mkdir(csv_path)
        except FileExistsError as e:
            print(e.strerror)
        bmp_list = os.listdir(path=target_path)
        print(bmp_list)
        for bmp in bmp_list:
            if find_by_pattern(bmp, pattern="train"):
                label.append(0)
            else:
                label.append(1)
            
        label_dict = dict(zip(bmp_list, label)) 
        print(label_dict)
        csv_file_path = os.path.join(csv_path, f'{name}.csv')
        try: 
            with open(csv_file_path, 'w') as f:
                w = csv.writer(f)
                for k,v in label_dict.items():
                    w.writerow([k,v])
        except OSError as e:
            print(e.strerror)
        #add_header
        df = pd.read_csv(csv_file_path, names=names)
        df.to_csv(csv_file_path, index=False)
        print("csv generation complete!")


    def clean_compressed_files(self):
        if "data" not in os.listdir(self.path):
            assert("no data directory here.")
        path = os.path.join(self.path, 'dataset')
        if os.path.isdir(path):
            dicts = os.listdir(path)
            for dict in dicts:
                if fnmatch(dict, "*zip"):
                    os.remove(Path(path, dict))
                if fnmatch(dict, "__MACOSX"):
                    os.system(f"rm -rf {Path(path, dict)}")

        # logger.info("verify if u need to clean the zip or rar files!")
        print("clean uncompressed files complete!")

    def wash_data(self):
        # path = os.path.join(self.path, "dataset")
        path = self.path
        temp_files_list = os.listdir(path)
        temp_files_list.remove("origin")
        temp_files_list.remove("test")
        temp_files_list.remove("csv")
        for tmp in temp_files_list:
            print(tmp)
            new_tmp = tmp.replace(" ",",")
            os.rename(os.path.join(path, tmp), os.path.join(path, new_tmp))
            os.system(f"rm -rf {os.path.join(path, new_tmp)}")
        
        # clean origin 
        # origin = os.path.join(path, "origin")
        # temp_files_list = os.listdir(origin)
        # for tmp in temp_files_list:
        #     if fnmatch(tmp, "[0-9]*.bmp"):
        #         os.remove(os.path.join(origin, tmp))

        print("clean completed!")

    def add_file_type_postfix(self):
        for old_name in self.dict:
            if fnmatch(old_name, "*-T") or fnmatch(old_name, "*-N"):
                new_name = old_name + ".dcm"
                os.rename(old_name, new_name)
                print(old_name, "===>", new_name)
        print("add postfix complete!")

    def convert_dcm_jpg_to_bmp(self, name):
        path = os.path.join(self.path, name)
        tmp_list = os.listdir(path)
        print(tmp_list)
        for tmp in tmp_list:
            if fnmatch(tmp, "*.dcm") or fnmatch(tmp, "*.jpg") :
                old_name = os.path.join(path, tmp)
                new_name = os.path.join(path, tmp[:-4])+".bmp"
            # if fnmatch(tmp, "*.tiff"):
            #     old_name = os.path.join(path, tmp)
            #     new_name = os.path.join(path, tmp[:-5])+".bmp"
                # print(new_name)
                os.system(f"dcmj2pnm --write-bmp {old_name} {new_name}")
                # os.remove(old_name)
        print("convertion complete!")

    def add_mask_frame(self):
        """
        for all origin files
        """
        
    
    
def find_by_pattern(filename, pattern="train"):
    if pattern == "train":
        s = r'.*-P0_.*'
        pattern = re.compile(s)
        if pattern.match(filename) is not None:
            return True
        return False       
    elif pattern == "N":
        s = r'.*-N..*'
        pattern = re.compile(s)
        if pattern.match(filename) is not None:
            return True
        return False
    elif pattern == "T":
        s = r'.*-T..*'
        pattern = re.compile(s)
        if pattern.match(filename) is not None:
            return True
        return False



if __name__ == '__main__':
    path = "../"
    preprocessor = DataPreprocessor(path=path)
    # unzip your data
    # preprocessor.uncompress_files()
    # preprocessor.clean_compressed_files()

    # all merge a datalist
    # preprocessor.all_merge_walk()
    # print(preprocessor.get_origin_datalist())

    # collect and clean
    target_path = "origin"
    # preprocessor.collect_datalist(target_path)
    # preprocessor.wash_data(target_path)
    # preprocessor.add_file_type_postfix()
    # preprocessor.convert_dcm_jpg_to_bmp()
    
    #generate_csv
    preprocessor.generate_csv(target_path)