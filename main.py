from SAC.Trainer import sacOnPolicyTrainer

if __name__ == '__main__':
    trainer = sacOnPolicyTrainer('./cfg/MacOS.json')
    trainer.run()