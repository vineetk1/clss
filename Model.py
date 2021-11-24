'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import LightningModule
import torch
from logging import getLogger
from sys import exit
from typing import Dict, List, Any
from importlib import import_module
import copy

logg = getLogger(__name__)


class Model(LightningModule):
    def __init__(self, model_init: dict = {}, app_specific_init: dict = {}):
        super().__init__()
        # save parameters for future use of "loading a model from
        # checkpoint" or "resuming training from checkpoint"
        self.save_hyperparameters()
        # Trainer('auto_lr_find': True,...) requires self.lr

        if model_init['model_type'] == "bert_large_uncased":
            from transformers import BertModel
            self.model = BertModel.from_pretrained('bert-large-uncased')
            class_head_dropout = model_init[
                'class_head_dropout'] if 'class_head_dropout' in model_init\
                else self.model.config.hidden_dropout_prob
            self.classification_head_dropout = torch.nn.Dropout(
                class_head_dropout)
            self.classification_head = torch.nn.Linear(
                self.model.config.hidden_size,
                app_specific_init['num_classes'])
            self.classification_head.weight.data.normal_(
                mean=0.0, std=self.model.config.initializer_range)
            if self.classification_head.bias is not None:
                self.classification_head.bias.data.zero_()
        else:
            strng = ('unknown model_type: ' f'{model_init["model_type"]}')
            logg.critical(strng)
            exit()

        if model_init['tokenizer_type'] == "bert":
            from transformers import BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained(
                'bert-large-uncased')
        else:
            strng = ('unknown tokenizer_type: '
                     f'{model_init["tokenizer_type"]}')
            logg.critical(strng)
            exit()

    def params(self, optz_sched_params, app_specific):
        self.app_specific = app_specific
        self.optz_sched_params = optz_sched_params
        # Trainer('auto_lr_find': True...) requires self.lr
        self.lr = self.optz_sched_params['optz_params'][
            'lr'] if 'lr' in self.optz_sched_params['optz_params'] else None

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self):
        logg.debug('')

    def training_step(self, batch: Dict[str, Any],
                      batch_idx: int) -> torch.Tensor:
        loss, _ = self.run_model(batch)
        # logger=True => TensorBoard; x-axis is always in steps=batches
        self.log('train_loss',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=False)
        return loss

    def training_epoch_end(self,
                           training_step_outputs: List[Dict[str,
                                                            torch.Tensor]]):
        avg_loss = torch.stack([x['loss']
                                for x in training_step_outputs]).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('train_loss_epoch', avg_loss,
                                          self.current_epoch)

    def validation_step(self, batch: Dict[str, Any],
                        batch_idx: int) -> torch.Tensor:
        loss, _ = self.run_model(batch)
        # checkpoint-callback monitors epoch val_loss, so on_epoch=True
        self.log('val_loss',
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=False)
        return loss

    def validation_epoch_end(self, val_step_outputs: List[torch.Tensor]):
        avg_loss = torch.stack(val_step_outputs).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('val_loss_epoch', avg_loss,
                                          self.current_epoch)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, logits = self.run_model(batch)
        # checkpoint-callback monitors epoch val_loss, so on_epoch=True
        self.log('test_loss_step',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        try:
            if self.statistics:
                self.statistics_step(example_ids=batch['sentence_ids'],
                                     predictions=torch.argmax(logits, dim=1),
                                     actuals=batch['labels'].squeeze(1))
        except AttributeError:
            pass
        return loss

    def test_epoch_end(self, test_step_outputs: List[torch.Tensor]):
        avg_loss = torch.stack(test_step_outputs).mean()
        # on TensorBoard, want to see x-axis in epochs (not steps=batches)
        self.logger.experiment.add_scalar('test_loss_epoch', avg_loss,
                                          self.current_epoch)
        try:
            if self.statistics:
                self.statistics_end()
        except AttributeError:
            pass

    def run_model(self, batch: Dict[str, Any]) -> torch.Tensor:
        outputs = self.model(**batch['model_inputs'])
        pooled_output = outputs[1]
        pooled_output = self.classification_head_dropout(pooled_output)
        logits = self.classification_head(pooled_output)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.app_specific['num_classes']),
                        batch['labels'].view(-1))
        return loss, logits
        # [0] => mean of losses from each example in batch; [1] => logits
        # return outputs[0], outputs[1]

    def configure_optimizers(self):
        opt_sch_params = copy.deepcopy(self.optz_sched_params)
        _ = opt_sch_params['optz_params'].pop('lr', None)
        if 'optz' in opt_sch_params and opt_sch_params['optz']:
            if 'optz_params' in opt_sch_params and opt_sch_params[
                    'optz_params']:
                if self.lr is not None:
                    optimizer = getattr(import_module('torch.optim'),
                                        opt_sch_params['optz'])(
                                            self.parameters(),
                                            lr=self.lr,
                                            **opt_sch_params['optz_params'])
                else:
                    optimizer = getattr(import_module('torch.optim'),
                                        opt_sch_params['optz'])(
                                            self.parameters(),
                                            **opt_sch_params['optz_params'])
            else:
                if self.lr is not None:
                    optimizer = getattr(import_module('torch.optim'),
                                        opt_sch_params['optz'])(
                                            self.parameters(), lr=self.lr)
                else:
                    optimizer = getattr(import_module('torch.optim'),
                                        opt_sch_params['optz'])(
                                            self.parameters())

        if 'lr_sched' in opt_sch_params and opt_sch_params['lr_sched']:
            if 'lr_sched_params' in opt_sch_params and opt_sch_params[
                    'lr_sched_params']:
                scheduler = getattr(import_module('torch.optim.lr_scheduler'),
                                    opt_sch_params['lr_sched'])(
                                        optimizer=optimizer,
                                        **opt_sch_params['lr_sched_params'])
            else:
                scheduler = getattr(
                    import_module('torch.optim.lr_scheduler'),
                    opt_sch_params['lr_sched'])(optimizer=optimizer)

        # If scheduler is specified then optimizer must be specified
        # If Trainer('resume_from_checkpoint',...), then optimizer and
        # scheduler may not be specified
        if 'optimizer' in locals() and 'scheduler' in locals():
            return {
                'optimizer':
                optimizer,
                'lr_scheduler':
                scheduler,
                'monitor':
                'val_loss'
                if opt_sch_params['lr_sched'] == 'ReduceLROnPlateau' else None
            }
        elif 'optimizer' in locals():
            return optimizer

    def clear_statistics(self):
        self.statistics = False

    def set_statistics(self, dataset_meta):
        self.statistics = True
        self.dataset_meta = dataset_meta
        num_classes = len(dataset_meta['class_info']['names'])
        self.confusion_matrix = torch.zeros(num_classes,
                                            num_classes,
                                            dtype=torch.int64)

    def statistics_step(self, example_ids, predictions, actuals):
        for prediction, actual in zip(predictions, actuals):
            self.confusion_matrix[prediction, actual] += 1

    def statistics_end(self):
        assert self.confusion_matrix.shape[0] == self.confusion_matrix.shape[1]
        epsilon = 1E-9
        precision = self.confusion_matrix.diag() / (
            self.confusion_matrix.sum(1) + epsilon)
        recall = self.confusion_matrix.diag() / (self.confusion_matrix.sum(0) +
                                                 epsilon)
        f1 = 2 * ((precision * recall) / (precision + recall + epsilon))
        f1_avg = f1.sum() / f1.shape[0]
        test_classes_lengths = torch.LongTensor(
            list(self.dataset_meta["class_info"]["test_lengths"].values()))
        f1_wgt = ((f1 * test_classes_lengths) /
                  test_classes_lengths.sum()).sum()

        print('\n\nAbout Dataset: original, train, validation, test')
        print(f'Split: N/A, {self.dataset_meta["dataset_info"]["split"]}')
        print(f'Lengths: {self.dataset_meta["dataset_info"]["lengths"]}')
        print(f'Batch_size: {self.dataset_meta["batch_size"]}')
        print(
            f'Steps per epoch: {[len/self.dataset_meta["batch_size"] for len in self.dataset_meta["dataset_info"]["lengths"]]}'
        )

        print('\nAbout Class distribution: original, train, validation, test')
        for prop in ['dataset_prop', 'train_prop', 'val_prop', 'test_prop']:
            if not self.dataset_meta["class_info"][prop]:
                print(' 0', end="")
            else:
                for num in list(
                        self.dataset_meta["class_info"][prop].values()):
                    print(f'{num: .4f}  ', end="")
            print('\n')

        print('About Test dataset:')
        for class_num, class_name in enumerate(
                self.dataset_meta['class_info']['names']):
            strng = (
                f'Class {class_num}, {class_name}, '
                f'{self.dataset_meta["class_info"]["test_lengths"][class_name]}'
                f' examples, '
                f'{self.dataset_meta["class_info"]["test_prop"][class_name]: .4f}'
                f' distribution')
            print(strng)

        print('\nconfusion matrix (prediction (rows) vs. actual (columns))=\n')
        print(f'{self.confusion_matrix}')
        print(f'precision = {precision}')
        print(f'recall = {recall}')
        print(f'F1 = {f1}')
        print(f'Average F1 = {f1_avg: .4f}')
        print(f'Weighted F1 = {f1_wgt: .4f}')
