from SAC.Trainer import sacOnPolicyTrainer

if __name__ == '__main__':
    trainer = sacOnPolicyTrainer('./cfg/MacOs.json')
    trainer.run()