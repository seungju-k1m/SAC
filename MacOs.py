from SAC.Trainer import sacTrainer

if __name__ == '__main__':
    trainer = sacTrainer('./cfg/v4img.json')
    trainer.run()