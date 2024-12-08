class Printer:
    def __init__(self, comps):
        self.comps = comps
        self.print_logo()
        self.print_train_details()
        self.debug = False
        
        
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
    
    def print_epoch_details(self, tr_score, tr_loss, vl_score, vl_loss):
        if not self.debug:
            return
        # print()
        print()
        print("\t Training - Score: ", tr_score, " Loss: ", tr_loss)
        print("\t Validation: - Score: ", vl_score, " Loss: ", vl_loss)
        print()
        
        
