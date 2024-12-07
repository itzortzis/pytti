import os
import time
import calendar

class Init:
    def __init__(self, components, params):
        self.parameters = params
        self.components = components
        # self.paths = paths
        print()
        
    def run(self):
        self.init_components()
        self.init_parameters()
        # self.init_paths()
        self.create_exp_dir()
        
    def init_components(self):
        self.model     = self.components['model']
        self.opt       = self.components['opt']
        self.loss_fn   = self.components['loss_fn']
        self.sets = self.components['sets']


    def init_parameters(self):
        self.classes    = self.parameters['classes']
        self.epochs     = self.parameters['epochs']
        self.epoch_thr  = self.parameters['epoch_thresh']
        self.score_thr  = self.parameters['score_thresh']
        self.device     = self.parameters['device']
        self.batch_size = self.parameters['batch_size']

    # def init_paths(self):
    #     self.trained_models = self.paths['trained_models']
    #     self.metrics = self.paths['metrics']
    #     self.figures = self.paths['figures']

    def create_exp_dir(self):
        current_GMT = time.gmtime()
        timestamp = calendar.timegm(current_GMT)
        exp_dir = "exp_" + str(timestamp)

        try:
            os.mkdir(exp_dir)
            os.mkdir(exp_dir + '/figures')
            os.mkdir(exp_dir + '/metrics')
            os.mkdir(exp_dir + '/inference')
        except OSError as error:
            print(error)

        self.exp_name = exp_dir
