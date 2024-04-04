from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import torch as t
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import copy

import datetime
 


from sklearn.model_selection import train_test_split


class Trainer:
    """
    Utility class for training and evaluating a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to be trained.
        crit (torch.nn.Module): The loss function for optimization.
        optim (torch.optim.Optimizer): The optimizer used for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler for controlling learning rate during training.
        train_dl (torch.utils.data.DataLoader): DataLoader for training data.
        val_dl (torch.utils.data.DataLoader): DataLoader for validation data.
        device (str): Device to be used for training (e.g., "cuda:0" for GPU or "cpu"). Defaults to "cuda:0".
        summary_writer_name (str): Name for the Tensorboard SummaryWriter.

    Attributes:
        _model (torch.nn.Module): The PyTorch model being trained.
        _crit (torch.nn.Module): The loss function for optimization.
        _optim (torch.optim.Optimizer): The optimizer used for training.
        _scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler for controlling learning rate during training.
        _train_dl (torch.utils.data.DataLoader): DataLoader for training data.
        _val_dl (torch.utils.data.DataLoader): DataLoader for validation data.
        _device (str): Device used for training (e.g., "cuda:0" for GPU or "cpu").
        _writer (torch.utils.tensorboard.SummaryWriter): SummaryWriter object for logging training progress.
    """

    def __init__(self,
                 model, 
                 crit,  
                 optim, 
                 scheduler,
                 train_dl,
                 val_dl,
                 device="cuda:0",
                 summary_writer_name=None):
          
        
        self._model = model
        self._crit = crit
        self._optim = optim
        self._scheduler = scheduler
        
        self._train_dl = train_dl
        self._val_dl = val_dl
        
        self._device = device
        if summary_writer_name == None:
            summary_writer_name = datetime.datetime.now()
        self._writer = SummaryWriter(f"runs/{summary_writer_name}")
        self._model = self._model.to(device)
        self._crit = self._crit.to(device)


    def train_step(self, x, y):
        """
        Perform a single training step by: 1) resetting the gradients, 2) predicting the class of a sample, 
        3) calculating the loss, 4) computing the gradient by backpropagation, 5) updating the waits,
        and 6) returning the loss

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target labels.

        Returns:
            torch.Tensor: Loss value for the training step.
        """
        self._model.zero_grad()
        outputs = self._model(x)        
        loss = self._crit(outputs, y)  
        loss.backward()
        self._optim.step()
        return loss

    def val_test_step(self, x, y):
        """
        Predict classification of sample and calculate loss (without training!)

        Args:
            x (torch.Tensor): Input data.
            y (torch.Tensor): Target labels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing loss value and model predictions.
        """
        outputs = self._model(x)
        loss = self._crit(outputs, y)
        return loss, outputs

    def train_epoch(self):
        """
        Train one epoch by 1) setting the training mode of the model, 2) calculating the average loss for each train step

        Returns:
            float: Average loss value for the epoch.
        """
        self._model.training = True
        self._model.train()
        
        losses = []
        train_dl_iter = iter(self._train_dl)
            
        while True:
            try:
                features, labels = next(train_dl_iter)
                features = features.to(self._device)
                labels = labels.to(self._device)
                losses.append(self.train_step(features, labels))
            except StopIteration:
                break   
        losses = t.tensor(losses)
        return t.mean(losses).numpy()

    def val_test(self):
        """
        Perform validation/test after one epoch.

        Returns:
            Tuple[float, float, float]: Tuple containing average loss, F1-score, and accuracy.
        """

        self._model.eval()
       
        losses = []
        f1_scores = []
        acc = []
        
        
        val_dl_iter = iter(self._val_dl)

        with t.no_grad():
            while True:
                try:
                    features, labels = next(val_dl_iter)
                    l = np.array(copy.deepcopy(labels))

                    features = features.to(self._device)
                    labels = labels.to(self._device)


                    loss, predictions = self.val_test_step(features, labels)
                    losses.append(loss)
                                        
                    predictions_np = predictions.detach().cpu().numpy()
                    
                    predictions_np = predictions_np > 0.5

                    f1_scores.append(f1_score(l, predictions_np.astype(int), average="macro"))
                    acc.append(accuracy_score(l, predictions_np.astype(int)))
                except StopIteration:
                    break
        losses = t.tensor(losses)
        return t.mean(losses).numpy(), np.mean(f1_scores), np.mean(acc)

    
    def train(self, epochs=50):
        """
        Train the model for a specified number of epochs.

        Args:
            epochs (int): Number of epochs to train the model. Defaults to 50.

        Returns:
            Tuple[list, list]: Tuple containing lists of training and validation losses.
        """

        assert epochs > 0
        
        train_losses = []
        val_losses = []
        
        e = 0

        maxacc = 0
        maxf1 = 0
        while True:
            if e == epochs:
                print("breaking")
                break
            

            train_loss = self.train_epoch()
            val_loss_ret = self.val_test()
            
            f1 = val_loss_ret[1]
            acc = val_loss_ret[2] 
            val_loss = val_loss_ret[0]
            
            self._scheduler.step(val_loss)
            if f1 > 0.7 or acc > 0.7:
                if f1 > maxf1 and acc > maxacc:
                    maxf1 = f1
                    maxacc = acc
                    t.save(self._model.state_dict(), f"model_{datetime.datetime.now()}_f1={f1}_acc={acc}_{e}.pt")
            
            
            self._writer.add_scalar(f"Train loss", train_loss, e)
            self._writer.add_scalar(f"Val loss", val_loss, e)
            self._writer.add_scalar(f"Val F1-score", f1, e)
            self._writer.add_scalar(f"Val Accuracy", acc, e)
            self._writer.add_scalar(f"LR", self._scheduler.get_last_lr()[0], e)
            self._writer.flush()
            
            """
            if e > self._early_stopping_patience:
                # if train_losses[len(train_losses) - 1] == train_losses[
                #     len(train_losses) - 2]:  # maybe use < and epsilon?
                #     break
                stop_early = True
                for i in range(e - self._early_stopping_patience, e):
                    if val_losses[i][0] > val_losses[e][0]:
                        stop_early = False
                        break
                if stop_early:
                    try:
                        self.save_checkpoint(e)
                    except:
                        pass
                    print(f"Stopping early: {val_losses}")
                    break
            """
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            e += 1
        return train_losses, val_losses
            
        
    def fit(self, epochs=-1):
        """
        Alias for the train method. Trains the model for a specified number of epochs.

        Args:
            epochs (int): Number of epochs to train the model. Defaults to -1.

        Returns:
            Tuple[list, list]: Tuple containing lists of training and validation losses.
        """
        finetune_losses, finetune_val_losses = self.train(epochs)
        self._writer.close()
        return finetune_losses, finetune_val_losses 