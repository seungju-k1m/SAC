from PPO.Trainer import PPOOnPolicyTrainer

if __name__ == '__main__':
    trainer = PPOOnPolicyTrainer('./cfg/v4.json')
    trainer.evaluate()
    print("Hello")