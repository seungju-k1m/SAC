from SAC.Trainer import sacTrainer

if __name__ == '__main__':
    trainer = sacTrainer('./cfg/MacOs.json')
    trainer.run()
    print("Hello")