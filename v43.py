from SAC.Trainer import sacTrainer

if __name__ == '__main__':
    trainer = sacTrainer('./cfg/v43.json')
    trainer.run()