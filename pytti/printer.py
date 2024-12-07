class Printer:
    def __init__(self, comps):
        self.comps = comps
        self.print_logo()
        self.print_train_details()
        
        
    def print_logo(self):
        print("PyTorch training-inference started")


    def print_train_details(self):
        self.print_logo()
        print('You are about to train the model on ' + self.comps.exp_name)
        print('with the following details:')
        print('\t Training epochs: ', self.comps.epochs)
        print('\t Epoch threshold: ', self.comps.epoch_thr)
        print('\t Score threshold: ', self.comps.score_thr)
        print('\t Device: ', self.comps.device)
        print()
        # option = input("Do you wish to continue? [Y/n]: ")
        return True or (option == 'Y' or option == 'y')
        
