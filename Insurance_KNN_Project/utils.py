import pickle
import json
import config
import numpy as np

class MedicalInsurance_KNN():
    def __init__(self,age,bmi,children,smoker,region):
        self.age = age
        self.bmi = bmi
        self.children = children
        self.smoker = smoker
        self.region = region
    def load_knn_model(self):
        with open(config.knn_pkl_path,'rb')as file:  ## Read KNN_Model.pkl file from Config
            self.pkl_knn_model = pickle.load(file)
        with open(config.knn_json_path,'r')as file:  ## Read knn_json file from Config
            self.json_knn_model = json.load(file)
        with open(config.knn_std_scaler_path,'rb')as file:        ## Read stdandardscaler file from Config
            self.std_scale_pkl = pickle.load(file)
    def get_charges(self):
        self.load_knn_model()
        test_array = np.zeros(len(self.json_knn_model['columns']))
        test_array[0] = self.children   #Go columns sequence wise as per json columns names...
        test_array[1] = self.json_knn_model['smoker'][self.smoker]
        region_1 = "region_" + self.region
        region_index = self.json_knn_model['columns'].index(region_1)
        test_array[region_index] = 1
        # if self.region == np.nan: ##Missing value ..if user has not enter any input
        #     self.region = 0
        test_array[6] = 0 if self.age<18 else 1 if self.age<30 else 2 if self.age<45 else 3 if self.age<60 else 4
        test_array[7] = 1 if self.age>60 else 0
        test_array[8] = 1 if self.bmi>25 else 0

        std_array = self.std_scale_pkl.transform([test_array]) #We pass test_array in 2D ,so it creates Dataframe 2D
        predict_charges = self.pkl_knn_model.predict(std_array)[0] #std_array already in 2D Dataframe...
        return np.around(predict_charges,2)
    

########### This file required in GitHub to push................>>>>>>>>>>>>>>>
# github 
# requirements.txt
# project files :  model .pkl, utils, app.py,config.py,std_scale, labeled.json
# frontend (HTML)

