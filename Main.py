'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from Data import Data
from Model import Model
from ast import literal_eval
from sys import argv
from logging import getLogger
from yaml import dump, full_load
import utils.logging_config
import collections.abc
from pathlib import Path

logg = getLogger(__name__)


def main():
    # last file name in command-line has dictionaries of parameters
    params_file_path = argv[len(argv) - 1]
    with open(params_file_path, 'r') as paramF:
        user_params = [
            dictionary for line in paramF if line[0] == '{'
            and isinstance(dictionary := literal_eval(line), dict)
        ]
    user_dicts_keys = [
        'misc', 'optz_sched', 'data', 'trainer', 'model_init',
        'ld_resume_chkpt'
    ]
    if len(user_params) != len(user_dicts_keys):
        strng = (f'{argv[1]} MUST have {len(user_dicts_keys)} '
                 f'dictionaries even if the dictionaries are empty.')
        logg.critical(strng)
        exit()
    user_params = {k: v for k, v in zip(user_dicts_keys, user_params)}
    if 'ld_chkpt' in user_params[
            'ld_resume_chkpt'] and 'resume_chkpt' in user_params[
                'ld_resume_chkpt']:
        logg.critical('Cannot load- and resume-checkpoint at the same time')
        exit()

    seed_everything(63)

    if 'ld_chkpt' in user_params['ld_resume_chkpt'] and user_params[
            'ld_resume_chkpt']['ld_chkpt'] is not None:
        model = Model.load_from_checkpoint(
            checkpoint_path=user_params['ld_resume_chkpt']['ld_chkpt'])
        dirPath = Path(user_params['ld_resume_chkpt']['ld_chkpt']).parents[1]
        chkpt_params = full_load(
            dirPath.joinpath('hyperparameters_used.yaml').read_text())
        # chkpt_params has 2 additional dicts - app_specific_init, app_specific
        assert len(user_params) == len(chkpt_params) - 2
        # override  some user_dicts with chkpt_dicts
        for user_dict_k in user_dicts_keys:
            if not user_params[user_dict_k] or user_dict_k == 'model_init':
                user_params[user_dict_k] = chkpt_params[user_dict_k]
        user_params['app_specific_init'] = chkpt_params['app_specific_init']
        user_params['app_specific'] = chkpt_params['app_specific']
        model.params(user_params['optz_sched'], user_params['app_specific'])
    elif 'resume_from_checkpoint' in user_params[
            'ld_resume_chkpt'] and user_params['ld_resume_chkpt'][
                'resume_from_checkpoint'] is not None:
        if 'resume_from_checkpoint' in user_params['trainer']:
            strng = (f'Remove "resume_from_checkpoint" from the "trainer" '
                     f'dictionary in the file {argv[1]}.')
            logg.critical(strng)
            exit()
        dirPath = Path(user_params['ld_resume_chkpt']
                       ['resume_from_checkpoint']).parents[1]
        chkpt_params = full_load(
            dirPath.joinpath('hyperparameters_used.yaml').read_text())
        # chkpt_params has 2 additional dicts - app_specific_init, app_specific
        assert len(user_params) == len(chkpt_params) - 2

        # override  some user_params with chkpt_params
        for (user_param_k,
             user_param_v), (chkpt_param_k,
                             chkpt_param_v) in zip(user_params.items(),
                                                   chkpt_params.items()):
            assert user_param_k == chkpt_param_k
            if not user_param_v or user_param_k == 'model_init' or\
                    user_param_k == 'optz_sched':
                user_params[user_param_k] = chkpt_param_v
        for user_dict_k in user_dicts_keys:
            if not user_params[user_dict_k] or user_dict_k == 'model_init' or\
                    user_dict_k == 'optz_sched':
                user_params[user_dict_k] = chkpt_params[user_dict_k]
        user_params['app_specific_init'] = chkpt_params['app_specific_init']
        user_params['app_specific'] = chkpt_params['app_specific']
        _ = user_params['trainer'].pop('resume_from_checkpoint', None)
        user_params['trainer']['resume_from_checkpoint'] = user_params[
            'ld_resume_chkpt']['resume_from_checkpoint']

        model = Model(user_params['model_init'],
                      user_params['app_specific_init'])
        model.params(user_params['optz_sched'], user_params['app_specific'])
    else:
        app_specific_init = {}  # parameters for initializing Model class
        app_specific = {}  # parameters needed throughout Model class
        app_specific_init['num_classes'] = 8
        app_specific['num_classes'] = 8
        user_params['app_specific_init'] = app_specific_init
        user_params['app_specific'] = app_specific
        model = Model(user_params['model_init'],
                      user_params['app_specific_init'])
        model.params(user_params['optz_sched'], user_params['app_specific'])
        tb_subDir = ",".join([
            f'{item}={user_params["model_init"][item]}'
            for item in ['model_type', 'tokenizer_type']
            if item in user_params['model_init']
        ])
        dirPath = Path('tensorboard_logs').joinpath(tb_subDir)
        dirPath.mkdir(parents=True, exist_ok=True)

    # create a directory to store all types of results
    new_version_num = max((int(dir.name.replace('version_', ''))
                           for dir in dirPath.glob('version_*')),
                          default=-1) + 1
    tb_logger = TensorBoardLogger(save_dir=dirPath,
                                  name="",
                                  version=new_version_num)
    dirPath = dirPath.joinpath('version_' + f'{new_version_num}')
    dirPath.mkdir(parents=True, exist_ok=True)
    paramFile = dirPath.joinpath('hyperparameters_used.yaml')
    paramFile.touch()
    paramFile.write_text(dump(user_params))

    # setup Callbacks and Trainer
    if not ('no_training' in user_params['misc']
            and user_params['misc']['no_training']):
        # Training: True, Testing: Don't care
        ckpt_filename = ""
        if 'batch_size' in user_params['data']:
            ckpt_filename += f'batch={user_params["data"]["batch_size"]},'
        for item in user_params['optz_sched']:
            if isinstance(user_params['optz_sched'][item], str):
                ckpt_filename += f'{item}={user_params["optz_sched"][item]},'
            elif isinstance(user_params['optz_sched'][item],
                            collections.abc.Iterable):
                for k, v in user_params['optz_sched'][item].items():
                    ckpt_filename += f'{k}={v},'
        ckpt_filename += '{epoch:02d}-{val_loss:.5f}'

        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=user_params['misc']['save_top_k']
            if 'save_top_k' in user_params['misc'] else 1,
            save_last=True,
            every_n_epochs=1,
            filename=ckpt_filename)
        lr_monitor = LearningRateMonitor(logging_interval='epoch',
                                         log_momentum=True)
        trainer = Trainer(logger=tb_logger,
                          deterministic=True,
                          num_sanity_val_steps=0,
                          log_every_n_steps=100,
                          callbacks=[checkpoint_callback, lr_monitor],
                          **user_params['trainer'])
    elif not ('no_testing' in user_params['misc']
              and user_params['misc']['no_testing']):
        # Training: False, Testing: True
        trainer = Trainer(logger=True,
                          checkpoint_callback=False,
                          **user_params['trainer'])
    else:
        # Training: False, Testing: False
        strng = ('User specified no-training and no-testing. Must do either'
                 'training or testing or both.')
        logg.critical(strng)
        exit()

    data = Data(user_params['data'])
    data.prepare_data(no_training=True if 'no_training' in user_params['misc']
                      and user_params['misc']['no_training'] else False,
                      no_testing=True if 'no_testing' in user_params['misc']
                      and user_params['misc']['no_testing'] else False)
    dataset_metadata = data.get_dataset_metadata()
    data.put_tokenizer(tokenizer=model.get_tokenizer())

    trainer.tune(model, datamodule=data)
    if not ('no_training' in user_params['misc']
            and user_params['misc']['no_training']):
        # Training: True, Testing: Don't care
        trainer.fit(model, datamodule=data)
        if not ('no_testing' in user_params['misc']
                and user_params['misc']['no_testing']):
            if 'statistics' in user_params['misc'] and user_params['misc'][
                    'statistics']:
                model.set_statistics(dataset_metadata)
            trainer.test()  # auto loads checkpoint file with lowest val loss
            model.clear_statistics()
    elif not ('no_testing' in user_params['misc']
              and user_params['misc']['no_testing']):
        # Training: False, Testing: True
        if 'statistics' in user_params['misc'] and user_params['misc'][
                'statistics']:
            model.set_statistics(dataset_metadata)
        trainer.test(model, test_dataloaders=data.test_dataloader())
        model.clear_statistics()
    else:
        # Training: False, Testing: False
        logg.critical('Bug in the Logic')
        exit()


if __name__ == '__main__':
    main()
