"""
reference:
https://blog.csdn.net/itplus/article/details/9361915
"""

import numpy as np
import random

# define reward function
# reward table is 5*5
R_tab = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 100],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]
], dtype=np.int32)

A_tab = {
    0: [4],
    1: [3, 5],
    2: [3],
    3: [1, 2, 4],
    4: [0, 3, 5],
    5: [1, 4, 5]
}

# print(Q)

Q_tab = np.zeros([6, 6], dtype=np.float32)

# parameters setting
alpha = 0.2
gamma = 0.2
epsilon = 0.2
episodes = 1000

# main loop
for episode in range(episodes):
    # choose random state
    current_state = random.randint(0, 5)
    while True:
        """
        # pure random
        action = random.choice(A_tab[current_state])
        """
        # epsilon-greedy
        if random.random() < epsilon:
            action = random.choice(A_tab[current_state])
        else:
            actionSet = A_tab[current_state]
            action = actionSet[0]
            maxVal = Q_tab[current_state][action]
            for ac in actionSet:
                if Q_tab[current_state][ac] > Q_tab[current_state][action]:
                    action = ac
        new_state = action
        # update Q_table
        Q_tab[current_state][action] = (1 - alpha) * Q_tab[current_state][action] + \
            alpha * (R_tab[current_state][action] + gamma * np.max(Q_tab[new_state]))
        # s = s'
        current_state = new_state
        # check condition
        if current_state == 5:
            break
    if episode % 100 == 0:
        print(f'episode: {episode}')
        print(Q_tab)

print("\ndone!\n")

# test result
route = []
for state in range(6):
    tmp = [state]
    while state != 5:
        action = np.argmax(Q_tab[state])
        tmp.append(action)
        state = action
    route.append(tmp)

print("The shortest route: ")
print(route)

"""
optimal route:
[[0, 4, 5], [1, 5], [2, 3, 1, 5]/[2, 3, 4, 5], [3, 1, 5]/[3, 4, 5], [4, 5], [5]]
"""
