import os
import shutil
import re
from fnmatch import fnmatch
import logging
from pathlib import Path
import csv
import pandas as pd

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

    def all_merge_walk(self):
        """
        os.walk tranversal all files in nested dirs 
        """
        base_path = self.path
        for root, dirs, files in os.walk(base_path):
            for name in files:
                if fnmatch(name, "*.bmp") or fnmatch(name, "*.jpg") or fnmatch(name, "*.dcm") or fnmatch(name, "A*") and not fnmatch(name, "*.docx"):
                    print(name)
                    self.dict.append(os.path.join(root, name))

    # def all_merge_(self):
    #     path = os.path.join(self.path, "data")
    #     lists1 = os.listdir(path)
    #     for list1 in lists1:
    #         #data/xx
    #         path1 = os.path.join(path, list1)
    #         if os.path.isdir(path1):
    #             lists2 = os.listdir(path1)
    #             for list2 in lists2:
    #                 #data/xx/A**
    #                 path2 = os.path.join(path1, list2)
    #                 #if there are no more dirs in remained part
    #                 if not os.path.isdir(path2):
    #                     if fnmatch(list2, "*.bmp") or fnmatch(list2, "*.jpg") or fnmatch(list2, "*.dcm") or fnmatch(list2, "A*") and not fnmatch(list2, "*.docx"):
    #                         print("find the bmp or jpg or dcn files.")
    #                         self.dict.append(path2)
    #                 else:
    #                     #go deeper
    #                     lists3 = os.listdir(path2)
    #                     for list3 in lists3:
    #                         path3 = os.path.join(path2, list3)
    #                         #like the pre loop ask if there is any dir
    #                         if not os.path.isdir(path3):
    #                             if fnmatch(list3, "*.bmp") or fnmatch(list3, "*.jpg") or fnmatch(list3, "*.dcm") or fnmatch(list3, "A*") and not fnmatch(list2, "*.docx"):
    #                                 print("find the bmp or jpg or dcm files")
    #                                 self.dict.append(path3)
    #                         else:
    #                             # go deeper
    #                             lists4 = os.listdir(path3)
    #                             for list4 in lists4:
    #                                 path4 = os.path.join(path3, list4)
    #                                 if not os.path.isdir(path4):
    #                                     if fnmatch(list4, "*.bmp") or fnmatch(list4, "*.jpg") or fnmatch(list4, "*.dcm") or fnmatch(list4, "A*") and not fnmatch(list2, "*.docx"):
    #                                         print("find the bmp or jpg or dcm files")
    #                                         self.dict.append(path4)
    #                                 else:
    #                                     # deepest 
    #                                     lists5 = os.listdir(path4)
    #                                     for list5 in lists5:
    #                                         path5 = os.path.join(path4, list5)
    #                                         if not os.path.isdir(path5):
    #                                              if fnmatch(list5, "*.bmp") or fnmatch(list5, "*.jpg") or fnmatch(list5, "*.dcm") or fnmatch(list5, "A*") and not fnmatch(list5, "*.docx"):
    #                                                 print("find the bmp or jpg or dcm files")
    #                                                 self.dict.append(path5)


    def collect_datalist(self):
        """
        collect data by self.dict(datapath list)
        """
        origin = os.path.join(self.path, "dataset", "origin")

        try:
            os.mkdir(origin)
        except OSError as e:    
            print(e.strerror)
        for sub_path in self.dict:
            try:
                shutil.move(sub_path, origin)
            except OSError as e:
                print(f'{sub_path}', e.strerror)


    def get_origin_datalist(self):
        if len(self.dict) == 0:
            return None
        return self.dict

    
  


    def generate_csv(self):
        path = os.path.join(self.path, 'dataset')
        names = ['img', 'label']
        label = []
        csv_path = os.path.join(path, 'csv')
        try:
            os.mkdir(csv_path)
        except FileExistsError as e:
            print(e.strerror)
        bmp_list = os.listdir(path=os.path.join(path, 'origin'))
        for bmp in bmp_list:
            if find_by_pattern(bmp):
                label.append(0)
            else:
                label.append(1)
        label_dict = dict(zip(bmp_list, label)) 
        print(label_dict)
        csv_file_path = os.path.join(csv_path, 'label.csv')
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
        path = os.path.join(self.path, "dataset")
        temp_files_list = os.listdir(path)
        temp_files_list.remove("origin")
        for tmp in temp_files_list:
            print(tmp)
            new_tmp = tmp.replace(" ",",")
            os.rename(os.path.join(path, tmp), os.path.join(path, new_tmp))
            os.system(f"rm -rf {os.path.join(path, new_tmp)}")
        
        # clean origin 
        origin = os.path.join(path, "origin")
        temp_files_list = os.listdir(origin)
        for tmp in temp_files_list:
            if fnmatch(tmp, "[0-9]*.bmp"):
                os.remove(os.path.join(origin, tmp))

        print("clean completed!")

    def add_file_type_postfix(self):
        for old_name in self.dict:
            if fnmatch(old_name, "*-T") or fnmatch(old_name, "*-N"):
                new_name = old_name + ".dcm"
                os.rename(old_name, new_name)
                print(old_name, "===>", new_name)
        print("add postfix complete!")

    def convert_dcm_jpg_to_bmp(self):
        path = os.path.join(self.path, "dataset", "origin")
        tmp_list = os.listdir(path)
        for tmp in tmp_list:
            if fnmatch(tmp, "*.dcm") or fnmatch(tmp, "*.jpg"):
                old_name = os.path.join(path, tmp)
                new_name = os.path.join(path, tmp[:-4])+".bmp"
                print(new_name)
                os.system(f"dcmj2pnm --write-bmp {old_name} {new_name}")
                os.remove(old_name)
        print("convertion complete!")

def find_by_pattern(filename):
    s = r'A\w*-\w*-\w*-P0_\w*-\w*-\w'
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
    # preprocessor.collect_datalist()
    # preprocessor.wash_data()
    # preprocessor.add_file_type_postfix()
    # preprocessor.convert_dcm_jpg_to_bmp()
    
    #generate_csv
    preprocessor.generate_csv()