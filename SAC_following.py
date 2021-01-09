from utils.SAC_Trainer import SAC_Trainer

if __name__ == '__main__':
    trainer = SAC_Trainer('./utils/SAC_config.json')

    trainer.run()
