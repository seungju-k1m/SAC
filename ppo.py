from PPO.Trainer import PPOOnPolicyTrainer

if __name__ == '__main__':
    trainer = PPOOnPolicyTrainer('./cfg/LSTMTest.json')
    trainer.evaluate()
    print("Hello")