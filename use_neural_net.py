import gym
import random
from tensorflow import keras
import numpy as np

if __name__ == '__main__':

    env = gym.make('LunarLander-v2')
    env.seed(0)
    np.random.seed(0)

    # Uses the trained network saved to file.
    model = keras.models.load_model('lunar_model.h5')

    # Constants
    episodes = 20 # Total number of game sto play.
    goal_steps = 500

    cumulative_score = 0
    best_score = 0

    for episode in range(episodes):        
        state = env.reset()
        state = np.reshape(state, (1, 8))
        score = 0

        for s in range(goal_steps):
            env.render()
            act_values = model.predict(state)
            action = np.argmax(act_values[0])
            
            next_state, reward, done, _ = env.step(action)
            score += reward
            cumulative_score += reward

            if done:
                break

            next_state = np.reshape(next_state, (1, 8))
            state = next_state

        if score > best_score:
            best_score = score

        average = cumulative_score / (episode + 1)
        print("Done episode: {}/{}, score: {}, average Score: {}\n"
            .format(episode + 1, episodes, score, average))

    print("Best score {}\n".format(best_score))