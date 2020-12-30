from PPO.Trainer import PPOOnPolicyTrainer

if __name__ == '__main__':
    trainer = PPOOnPolicyTrainer('./cfg/v2.json')
    trainer.run()
    print("Hello")