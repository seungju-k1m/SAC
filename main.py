import ray
import os
import argparse
# import numpy as np
from PPO.Trainer import PPOOnPolicyTrainer


parser = argparse.ArgumentParser(
    description="훈련 파일입니다.")
parser.add_argument(
    '--path',
    '-p',
    type=str,
    default='./cfg/MLPTrain.json',
    help="The relative file path of cfg.json")
parser.add_argument(
    '--train',
    '-t',
    action='store_true',
    default=False,
    help="set train mode"
)

parser.add_argument(
    '--test',
    '-te',
    action='store_true',
    default=False,
    help="set test mode"
)
args = parser.parse_args()


if __name__ == '__main__':
    # mp.set_start_method('spawn')
    ray.init(num_cpus=8, local_mode=True)
    path = args.path
    if os.path.isfile(path):
        print("Finish Loading Configuration File.")
    else:
        RuntimeError("There is no file in path. You must checkt the file location")
    
    if path[:-5] != '.json':
        RuntimeError("The format of file is not .json")

    test = args.test
    train = args.train
    trainer = PPOOnPolicyTrainer(path)
    if train:
        print("""
        ------------------------------------------------
        Train Mode
        ------------------------------------------------
        """)
        trainer.run()
    if test:
        print("""
        ------------------------------------------------
        Test Mode
        ------------------------------------------------
        """)
        trainer.evaluate()
    
    trainer.writeTrainInfo()