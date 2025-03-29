import time

import pytorch_lightning as pl

import torch
import numpy as np
import torch.nn as nn
import src.constants as cst
from src.metrics.metrics_learning import compute_metrics


class LOBCAST_NNEngine(pl.LightningModule):
    def __init__(self, neural_architecture, loss_weights, hps, metrics_log, wandb_log):
        super().__init__()
        self.neural_architecture = neural_architecture
        self.loss_weights = loss_weights
        self.hps = hps
        self.metrics_log = metrics_log
        self.wandb_log = wandb_log
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def log_wandb(self, metrics):
        if self.wandb_log:
            self.wandb_log.log(metrics)

    def forward(self, batch):
        # time x features - 40 x 100 in general
        out = self.neural_architecture(batch)
        logits = nn.Softmax(dim=1)(out)  # todo check if within model
        return out, logits

    def training_step(self, batch, batch_idx):
        prediction_ind, y, loss_val, logits = self.make_predictions(batch)
        self.training_step_outputs.append((prediction_ind, y, loss_val, logits))
        return {"loss": loss_val, "other": (prediction_ind, y, loss_val, logits)}

    def validation_step(self, batch, batch_idx):
        prediction_ind, y, loss_val, logits = self.make_predictions(batch)
        self.validation_step_outputs.append((prediction_ind, y, loss_val, logits))
        return prediction_ind, y, loss_val, logits

    def test_step(self, batch, batch_idx):
        prediction_ind, y, loss_val, logits = self.make_predictions(batch)
        self.test_step_outputs.append((prediction_ind, y, loss_val, logits))
        return prediction_ind, y, loss_val, logits

    def make_predictions(self, batch):
        x, y = batch
        out, logits = self(x)
        loss_val = nn.CrossEntropyLoss(self.loss_weights)(out, y)

        # deriving prediction from softmax probs
        prediction_ind = torch.argmax(logits, dim=1)  # B
        return prediction_ind, y, loss_val, logits

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        t0 = time.time()
        self(x)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        print("Inference for the model:", elapsed, "ms")
        return elapsed

    def evaluate_classifier(self, stp_type, step_outputs):
        preds, truths, loss_vals, logits = self.__get_prediction_vectors(step_outputs)
        eval_dict = compute_metrics(truths, preds, loss_vals)

        var_name = "{}_{}".format(stp_type, cst.Metrics.LOSS.value)
        self.log(var_name, eval_dict[cst.Metrics.LOSS.value], prog_bar=True)

        var_name = "{}_{}".format(stp_type, cst.Metrics.F1.value)
        self.log(var_name, eval_dict[cst.Metrics.F1.value], prog_bar=True)

        path = cst.METRICS_BEST_FILE_NAME if self.metrics_log.is_best_model else cst.METRICS_RUNNING_FILE_NAME

        print("\n")
        print(f"END epoch {self.current_epoch} ({stp_type})")
        print("Logging stats...")
        self.metrics_log.add_metric(self.current_epoch, stp_type, eval_dict)
        self.metrics_log.dump_metrics(path)
        self.log_wandb({f"{stp_type}_{k}": v for k, v in eval_dict.items()})
        print("Done.")



        # Step 3: Replace training_epoch_end with on_train_epoch_end
    def on_train_epoch_end(self):
        # Process the collected outputs
        outputs = self.training_step_outputs
        print("Outputs type:", type(outputs))
        if outputs:  # if outputs is not empty
            print("First output type:", type(outputs[0]))
            print("First output content:", outputs[0])

        # Put your existing training_epoch_end logic here
        #training_step_outputs = [batch["other"] for batch in outputs]
        self.evaluate_classifier(cst.ModelSteps.TRAINING.value, outputs)
        # Clear the outputs list for the next epoch
        self.training_step_outputs = []


    def on_validation_epoch_end(self):
        # Process the collected outputs
        outputs = self.validation_step_outputs
        # Put your existing validation_epoch_end logic here
        self.evaluate_classifier(cst.ModelSteps.VALIDATION.value, outputs)
        # Clear the outputs list for the next epoch
        self.validation_step_outputs = []


    def on_test_epoch_end(self):
        # Process all outputs here
        outputs = self.test_step_outputs
        self.evaluate_classifier(cst.ModelSteps.TESTING.value, outputs)
        # Clear the outputs list for the next epoch
        self.test_step_outputs.clear()

    def __get_prediction_vectors(self, model_output):
        """ Accumulates the models output after each validation and testing epoch end. """

        preds, truths, losses, logits = [], [], [], []
        for preds_b, y_b, loss_val, logits_b in model_output:
            preds += preds_b.tolist()
            truths += y_b.tolist()
            logits += logits_b.tolist()
            losses += [loss_val.item()]  # loss is single per batch

        preds  = np.array(preds)
        truths = np.array(truths)
        logits = np.array(logits)
        losses = np.array(losses)

        return preds, truths, losses, logits

    def configure_optimizers(self):
        if self.hps.OPTIMIZER == "SGD":
            return torch.optim.SGD(self.parameters(), lr=self.hps.LEARNING_RATE)
        elif self.hps.OPTIMIZER == "ADAM":
            return torch.optim.Adam(self.parameters(), lr=self.hps.LEARNING_RATE)
        elif self.hps.OPTIMIZER == "RMSPROP":
            return torch.optim.RMSprop(self.parameters(), lr=self.hps.LEARNING_RATE)
