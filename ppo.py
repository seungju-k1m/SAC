from PPO.Trainer import PPOOnPolicyTrainer

if __name__ == '__main__':
    trainer = PPOOnPolicyTrainer('./cfg/MacOSPPOLSTM.json')
    trainer.run()
    print("Hello")