import gym
import gym_tetris

from statistics import mean, median
from gym_tetris.ai.QNetwork import QNetwork


def main():
    env = gym.make("tetris-v1", action_mode=1)
    network = QNetwork()
    network.load()
    episodes=25

    running = True
    total_games = 0
    total_steps = 0
    s = []
    while running:
        steps, rewards, scores = network.train(env, episodes=episodes)
        total_games += len(scores)
        total_steps += steps
        network.save()
        print("==================")
        print("* Total Games: ", total_games)
        print("* Total Steps: ", total_steps)
        print("* Epsilon: ", network.epsilon)
        print("*")
        print("* Average: ", sum(rewards) / len(rewards), "/", sum(scores) / len(scores))
        print("* Median: ", median(rewards), "/", median(scores))
        print("* Mean: ", mean(rewards), "/", mean(scores))
        print("* Min: ", min(rewards), "/", min(scores))
        print("* Max: ", max(rewards), "/", max(scores))
        print("==================")
        
        s.append(mean(scores))
        if total_games >= 12500:
            break

    env.close()
    
    import matplotlib.pyplot as plt
    
    plt.plot([i*25+25 for i in range(len(s))], s)
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.title("Avg. Training Rewards")
    plt.savefig('plot.png')
    
if __name__ == '__main__':
    main()
