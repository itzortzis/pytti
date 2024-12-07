
import time
import torch
import calendar
import numpy as np
from tqdm import tqdm
from torchmetrics import F1Score
from sklearn.metrics import accuracy_score, f1_score



from pytti.pre_processing import DataPreprocesing
from pytti.initialization import Init
from pytti.printer import Printer
from pytti.metrics import Metrics


class Training():

    def __init__(self, comps, params):
        self.comps = Init(comps, params)
        self.comps.run()
        self.init()


    def init(self):
        self.pr = Printer(self.comps)
        self.dpp = DataPreprocesing(self.comps)
        self.dpp.run()
        self.metrics = Metrics(self.comps)

        if self.comps.device == 'cuda':
            print("Cuda available")
            self.comps.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.comps.model = self.comps.model.to(self.comps.device)




    # Main_training:
    # --------------
    # The supervisor of the training procedure.
    def main_training(self):
        if not (self.pr.print_train_details()):
            return
        self.metrics.init_metrics()
        for epoch in tqdm(range(self.comps.epochs)):
            
            tr_score, tr_loss = self.epoch_training()
            vl_score, vl_loss = self.epoch_validation()

            self.metrics.losses[epoch, 0] = tr_loss
            self.metrics.losses[epoch, 1] = vl_loss
            self.metrics.scores[epoch, 0] = tr_score
            self.metrics.scores[epoch, 1] = vl_score

            print()
            print("\t Training - Score: ", tr_score, " Loss: ", tr_loss)
            print("\t Validation: - Score: ", vl_score, " Loss: ", vl_loss)
            print()
            self.metrics.save_model_weights(epoch, vl_score, self.comps.model.state_dict())
            self.comps.model_dict = self.comps.model.state_dict()
        self.metrics.calculate_exec_time()
        self.metrics.test_set_score = self.inference()
        self.metrics.update_log()


    


    


    

    # Prepare_data:
    # -------------
    # Given x and y tensors, this function applies some basic
    # transformations/changes related to dimensions, data types,
    # and device.
    #
    # --> x: tensor containing a batch of input images
    # --> y: tensor containing a batch of annotation masks
    # <-- x, y: the updated tensors
    def prepare_data(self, x, y):
        x = torch.unsqueeze(x, 1)

        x = x.to(torch.float32)
        y = y.to(torch.int64)

        x = x.to(self.comps.device)
        y = y.to(self.comps.device)

        return x, y
        
  
    def argmax_ys(self, labels, preds):
        s_preds = torch.softmax(preds, dim=0)
        a_preds = torch.argmax(preds, dim=0)
        a_labels = torch.argmax(labels, dim=0)

        return s_preds, a_preds, a_labels

    # Epoch_training:
    # ---------------
    # This function is used for implementing the training
    # procedure during a single epoch.
    #
    # <-- epoch_score: performance score achieved during
    #                  the training
    # <-- epoch_loss: the loss function score achieved during
    #                 the training
    def epoch_training(self):
        self.comps.model.train(True)
        current_score = 0.0
        current_loss = 0.0
        self.metrics.f1.reset()
        # print("Simple epoch training...")

        step = 0
        for x, y in self.dpp.train_ldr:
            x, y = self.prepare_data(x, y)
            step += 1
            self.comps.opt.zero_grad()
            outputs = self.comps.model(x)
            loss = self.comps.loss_fn(outputs, y.float())
            loss.backward()
            self.comps.opt.step()
            preds = torch.argmax(outputs, dim=1)
            labels = torch.argmax(y, dim=1)
            # print(preds,labels)
            score = self.metrics.f1.update(preds, labels)
            current_score += f1_score(preds.cpu().detach().numpy(), labels.cpu().detach().numpy(), average='weighted')
            current_loss  += loss * self.dpp.train_ldr.batch_size

        epoch_score = current_score/len(self.dpp.train_ldr)
        # epoch_score = self.metrics.f1.compute()
        self.metrics.f1.reset()
        epoch_loss  = current_loss / len(self.dpp.train_ldr.dataset)

        return epoch_score, epoch_loss.item()

    
 
    # Epoch_validation:
    # ---------------
    # This function is used for implementing the validation
    # procedure during a single epoch.
    #
    # <-- epoch_score: performance score achieved during
    #                  the validation
    # <-- epoch_loss: the loss function score achieved during
    #                 the validation
    def epoch_validation(self):
        self.comps.model.train(False)
        current_score = 0.0
        current_loss = 0.0
        self.metrics.f1.reset()
        for x, y in self.dpp.valid_ldr:
            x, y = self.prepare_data(x, y)

            with torch.no_grad():
                outputs = self.comps.model(x)
            m = torch.nn.Sigmoid()
            loss = self.comps.loss_fn(outputs, y.float())
            preds = torch.argmax(outputs, dim=1)
            labels = torch.argmax(y, dim=1)
            score = self.metrics.f1.update(preds, labels)
            current_score += f1_score(preds.cpu().detach().numpy(), labels.cpu().detach().numpy(), average='weighted')
            current_loss  += loss * self.dpp.train_ldr.batch_size
        epoch_score = current_score/len(self.dpp.valid_ldr)
        # epoch_score = self.metrics.f1.compute()
        epoch_loss  = current_loss / len(self.dpp.valid_ldr.dataset)

        return epoch_score, epoch_loss.item()


    # Inference:
    # ----------
    # Applies inference to the testing set extracted from
    # the input dataset during the initialization phase
    #
    # <-- test_set_score: the score achieved by the trained model
    def inference(self):
        self.comps.model.load_state_dict(self.comps.model_dict)
        self.comps.model.eval()
        current_score = 0.0
        current_loss = 0.0
        self.metrics.f1.reset()
        for x, y in self.dpp.test_ldr:
            x, y = self.prepare_data(x, y)

            with torch.no_grad():
                outputs = self.comps.model(x)

            preds = torch.argmax(outputs, dim=1)
            labels = torch.argmax(y, dim=1)
            score = self.metrics.f1.update(preds, labels)
            current_score += f1_score(preds.cpu().detach().numpy(), labels.cpu().detach().numpy(), average='weighted')

        test_set_score = current_score/len(self.dpp.test_ldr)
        # test_set_score = self.metrics.f1.compute()
        self.metrics.f1.reset()
        return test_set_score.item()


    def ext_inference(self, set_ldr):
        path_to_model = self.comps.trained_models + self.comps.inf_model
        self.comps.model.load_state_dict(torch.load(path_to_model))
        self.comps.model.eval()
        self.f1 = F1Score(task="multiclass", num_classes=self.comps.classes)
        self.f1.to(self.comps.device)
        current_score = 0.0
        # self.f1.reset()
        for x, y in set_ldr:
            x,  y = self.prepare_data(x, y)

            with torch.no_grad():
                outputs = self.comps.model(x)
            preds = torch.argmax(outputs, dim=1)
            labels = torch.argmax(y, dim=1)
            # preds = torch.argmax(outputs, dim=1)
            current_score += f1_score(preds.cpu().detach().numpy(), labels.cpu().detach().numpy(), average='weighted')
            self.f1.update(preds, labels)

        inf_score = current_score/len(set_ldr)
        
        # inf_score = self.f1.compute()
        self.f1.reset()
        return inf_score.item()
