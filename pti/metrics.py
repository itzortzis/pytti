import time
import torch
import calendar
import numpy as np
from torchmetrics import F1Score
from torchmetrics.classification import BinaryF1Score
from matplotlib import pyplot as plt



class Metrics:
    def __init__(self, comps):
        self.comps = comps
        self.losses = np.zeros((self.comps.epochs, 2))
        self.scores = np.zeros((self.comps.epochs, 2))
        self.max_score = 0
        self.log_line = ""
        self.log = open("logs.txt", "a")  # append mode
        self.test_set_score = 0.0
        
        
    def init_metrics(self):
        print("Initializing metrics")
        self.f1 = F1Score(task="multiclass", num_classes=5)
        # self.f1 = BinaryF1Score()
        self.f1.to(self.comps.device)
        self.get_current_timestamp()
        self.start_time = time.time()
        
        
    # Get_current_timestamp:
    # ----------------------
    # This function calculates the current timestamp that is
    # used as unique id for saving the experimental details
    def get_current_timestamp(self):
        current_GMT = time.gmtime()
        self.timestamp = calendar.timegm(current_GMT)
        
    
    def calculate_exec_time(self):
        self.exec_time = time.time() - self.start_time
        
        
    # Save_model_weights:
    # -------------------
    # This funtion saves the model weights during training
    # procedure, if some requirements are satisfied.
    #
    # --> epoch: current epoch of the training
    # --> score: current epoch score value
    # --> loss: current epoch loss value
    def save_model_weights(self, epoch, score, model_state_dict):

        if score > self.max_score and epoch > self.comps.epoch_thr:
            path_to_model = self.comps.trained_models + self.comps.exp_name
            path_to_model += "_" + str(self.timestamp) + ".pth"
            torch.save(model_state_dict, path_to_model)
            log = str(epoch) + " " + str(score) + " " + path_to_model + "\n"
            self.log_line = log
            self.max_score = score
            
            
    def save_metrics(self):
        postfix = self.comps.exp_name + "_" + str(self.timestamp)
        np.save(self.comps.metrics + "scores_" + postfix, self.scores)
        np.save(self.comps.metrics + "losses_" + postfix, self.losses)
        self.save_figures()
        
    
    def save_figures(self):
        postfix = self.comps.exp_name + "_" + str(self.timestamp) + ".png"
        plt.figure()
        plt.plot(self.scores[:, 0])
        plt.savefig(self.comps.figures + "train_s_" + postfix)
        plt.figure()
        plt.plot(self.scores[:, 1])
        plt.savefig(self.comps.figures + "valid_s_" + postfix)

        plt.figure()
        plt.plot(self.losses[:, 0])
        plt.savefig(self.comps.figures + "train_l_" + postfix)
        plt.figure()
        plt.plot(self.losses[:, 1])
        plt.savefig(self.comps.figures + "valid_l_" + postfix)


    def update_log(self):
        self.log_line = str(self.test_set_score) + " " + self.log_line
        self.save_metrics()
        self.log.write(self.log_line)
        self.log.close()
        print("Total execution time: ", self.exec_time, " seconds")
