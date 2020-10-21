from SAC.Trainer import sacTrainer

if __name__ == '__main__':
    trainer = sacTrainer('./cfg/01_SAC.json')
    trainer.run()