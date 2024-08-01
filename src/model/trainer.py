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
        self._summary_writer_name = summary_writer_name
        self._writer = SummaryWriter(f"runs/{summary_writer_name}")
        self._model = self._model.to(device)
        self._crit = self._crit.to(device)


    def train_step(self, x, y, epoch):
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
        outputs = self._model(x, epoch)    
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
        binary = outputs > 0.5
        binary_loss = self._crit(binary.float(), y)
        return loss, outputs, binary_loss

    def train_epoch(self, epoch):
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
                features, labels, _ = next(train_dl_iter)
                features = features.to(self._device)
                labels = labels.to(self._device)
                losses.append(self.train_step(features, labels, epoch=epoch))
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
        binary_losses = []
        f1_scores = []
        acc = []
        batch_ids = list()
        batch_predictions = list()
        batch_labels = list()
        
        
        val_dl_iter = iter(self._val_dl)

        with t.no_grad():
            while True:
                try:
                    features, labels, ids = next(val_dl_iter)
                    l = np.array(copy.deepcopy(labels))

                    features = features.to(self._device)
                    labels = labels.to(self._device)


                    loss, predictions, binary_loss = self.val_test_step(features, labels)
                    losses.append(loss)
                    binary_losses.append(binary_loss)
                                        
                    predictions_np = predictions.detach().cpu().numpy()
                    predictions_np = predictions_np > 0.5
                    
                    batch_ids.append(np.array(ids))
                    batch_predictions.append(predictions_np)
                    batch_labels.append(l)
                    #if np.sum(l[pos_labels]) > 0 and np.sum(np.invert(l.astype(bool))[neg_labels]) > 0:
                    #    pos_labels = np.where(l == 1)
                    #    pos_acc = np.sum(predictions_np[pos_labels]) / np.sum(l[pos_labels])
                    #    neg_labels = np.where(l == 0)
                    #    neg_acc = np.sum(np.invert(predictions_np.astype(bool))[neg_labels]) / np.sum(np.invert(l.astype(bool))[neg_labels])
                    #    acc.append(np.mean([pos_acc, neg_acc]))
                        
                    #else:
                    acc.append(accuracy_score(l, predictions_np.astype(int)))
                    f1_scores.append(f1_score(l, predictions_np.astype(int), average="weighted"))
                    
                except StopIteration:
                    break

        batch_ids = np.concatenate(batch_ids)
        batch_predictions = np.concatenate(batch_predictions)
        batch_labels = np.concatenate(batch_labels)
        
        majority_acc = list()
        
        for id in np.unique(batch_ids):
            label = np.mean(batch_labels[np.where(batch_ids == id)]) >= 0.5
            prediction = np.mean(batch_predictions[np.where(batch_ids == id)]) >= 0.5
            majority_acc.append((label == prediction))
        losses = t.tensor(losses)
        return t.mean(losses).numpy(), np.mean(f1_scores), np.mean(acc), np.mean(majority_acc)

    def write_val(self, f1, acc, val_loss, majority_acc, e):
        self._writer.add_scalar(f"Val loss", val_loss, e)
        self._writer.add_scalar(f"Val F1-score", f1, e)
        self._writer.add_scalar(f"Val Accuracy", acc, e)
        self._writer.add_scalar(f"Val Majority Accuracy", majority_acc, e)

        return
        
    
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
        
            train_loss = self.train_epoch(epoch=e)
            if self._val_dl:
                val_loss, f1, acc, majority_acc = self.val_test()
                self._scheduler.step()
                self.write_val(f1, acc, val_loss, majority_acc, e)
                val_losses.append(val_loss)

            self._writer.add_scalar(f"Train loss", train_loss, e)
            self._writer.add_scalar(f"LR", self._scheduler.get_last_lr()[0], e)
            self._writer.flush()
            train_losses.append(train_loss)
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