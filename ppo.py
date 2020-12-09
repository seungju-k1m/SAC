from PPO.Trainer import PPOOnPolicyTrainer

if __name__ == '__main__':
    trainer = PPOOnPolicyTrainer('./cfg/MacOsPPOLSTM.json')
    trainer.run()
    print("Hello")