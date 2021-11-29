# Classification using Deep Learning
## Requirements
* PyTorch version >= 1.9.1+cu111
* Python version >= 3.8.10
* PyTorch-Lightning version >= 1.4.9
* Huggingface Transformers version >= 4.11.3
* Tensorboard version >= 2.6.0
* Pandas >= 1.3.4
* Scikit-learn: numpy>=1.14.6, scipy>=1.1.0, threadpoolctl>=2.0.0, joblib>=0.11
## Installation
```
pip3 install transformers
pip3 install pytorch-lightning
pip3 install tensorboard
pip3 install pandas
pip3 install scikit-learn
git clone https://github.com/vineetk1/clss.git
cd clss
```
Note that the default directory is *clss*. Unless otherwise stated, all commands from the Command-Line-Interface must be delivered from the default directory.
## Download the dataset
1. Make a *data* directory.      
```
mkdir data
```
2. Download a dataset in the *data* directory.       
## Saving all informtion and results of an experiment
All information about the experiment is stored in a unique directory whose path starts with tensorboard_logs and ends with a unique version-number. Its contents consist of hparams.yaml, hyperperameters_used.yaml, test-results.txt, events.* files, and a checkpoints directory that has one or more checkpoint-files.
## Train, validate, and test a model
Following command trains a model, saves the last checkpoint plus checkpoints that have the lowest validation loss, runs the test dataset on the checkpointed model with the lowest validation loss, and outputs the results of the test:
```
python3 Main.py input_param_files/bert_seq_class
```
The user-settable hyper-parameters are in the file *input_param_files/bert_seq_class*. An explanation on the contents of this file is at *input_param_files/README.md*. A list of all the hyper-parameters is in the <a href="https://www.pytorchlightning.ai" target="_blank">PyTorch-Lightning documentation</a>, and any hyper-parameter can be used.    
To assist in Training, the two parameters *auto_lr_find* and *auto_scale_batch_size* in the file *input_param_files/bert_seq_class* enable the software to automatically find an initial Learning-Rate and a Batch-Size respectively.    
As training progresses, graphs of *"training-loss vs. epoch #"*, *"validation-loss vs. epoch #"*, and "learning-rate vs. batch #" are plotted in real-time on the TensorBoard.  Training is stopped by typing, at the Command-Line-Interface, the keystroke ctrl-c. The current training information is checkpointed, and training stops. Training can be resumed, at some future time, from the checkpointed file.   
Dueing testing, the results are sent to the standard-output, and also saved in the *test-results.txt" file that include the following: general information about the dataset and the classes, confusion matrix, precision, recall, f1, average f1, and weighted f1.
## Resume training, validation, and testing a model with same hyper-parameters
Resume training a checkpoint model with the same model- and training-states by using the following command:
```
python3 Main.py input_param_files/bert_seq_class-res_from_chkpt
```
The user-settable hyper-parameters are in the file *input_param_files/bert_seq_class-res_from_chkpt*.  An explanation on the contents of this file is at *input_param_files/README.md*.
## Change hyper-parameters and continue training, validation, and testing a model
Continue training a checkpoint model with the same model-state but different hyperparameters for the training-state by using the following command:
```
python3 Main.py input_param_files/bert_seq_class-ld_chkpt
```
The user-settable hyper-parameters are in the file *input_param_filesbert_seq_class-ld_chkpt*.  An explanation on the contents of this file is at *input_param_files/README.md*.   
## Further test a checkpoint model with a new dataset
1. Download a new dataset in the *data* directory.    
1. Locate the checkpoint model to use for testing with a new dataset. The path is of a following form: tensorboard_logs/model_type=bert_seqClassification_large_uncased,tokenizer_type=bert/version_0/checkpoints/batch=10,optz=Adam,lr=2e-05,lr_sched=ReduceLROnPlateau,mode=min,patience=1,factor=0.1,epoch=02-val_loss=0.39129.ckpt   
    1. The most recent trained model is in a subdirectory that has the highest version number. For example, version_0 is the first (i.e. oldest) trained model, version_1 is the second trained model, and so on.
    1. The name of a checkpoint file includes the validation-loss. Pick the file that has the lowest validation loss (e.g. val_loss=0.39129).
1. In the *input_param_files/bert_seq_class-test_only* file, replace the path of the checkpoint file.
```
{..............., 'chkpt': 'tensorboard_logs/model_type=bert_seqClassification_large_uncased,tokenizer_type=bert/version_0/checkpoints/batch=10,optz=Adam,lr=2e-05,lr_sched=ReduceLROnPlateau,mode=min,patience=1,factor=0.1,epoch=02-val_loss=0.39129.ckpt'}   
```    
4. In the *input_param_files/bert_seq_class-test_only* file, replace the name of the dataset file.
```
{'default_format_path': 'data/test_file.csv', ............} 
```
5. Following command, on the command-line, tests the model with the new dataset:
```
python3 Main.py input_param_files/bert_seq_class-test_only 
```
