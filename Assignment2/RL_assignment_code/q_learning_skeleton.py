import numpy as np
import random
NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500

T = 3# Temperature for boltzman eqautions
T_DECAY = 0.9
DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1
VI_THRESHOLD = 0.0005



class QLearner():
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE, exploration_strat = 'egreedy'):
        self.name = "agent1"
        self.discount = discount
        self.learning_rate = learning_rate
        self.num_states = num_states
        self.num_actions = num_actions
        self.episode_durrations = []
        self.q = np.zeros((num_states, num_actions))
        self.strategey = exploration_strat
        self.pi = np.ones((num_states, num_actions))

    def reset_episode(self, episode_duration):

        """
        Here you can update some of the statistics that could be helpful to maintain
        """
        self.episode_durrations.append(episode_duration)
        pass

    def process_experience(self, state, action, next_state, reward, done): 
        """
        Update the Q-value based on the state, action, next state and reward.
        """

        update = (1-self.learning_rate)*self.q[state,action] + self.learning_rate*reward

        if not done:
            update += self.learning_rate*self.discount*np.max(self.q[next_state][:])
        
        self.q[state,action] = update

    def select_action(self, state):

        if self.strategey == 'boltz':
            self.pi = np.exp(self.q / max((T*(T_DECAY**len(self.episode_durrations))), 0.03))
            sum_rows = np.sum(self.pi, axis=1)
            self.pi = (self.pi.T / sum_rows).T
            return np.random.choice(range(self.num_actions), p= self.pi[state])

        if self.strategey == 'egreedy':
            selection = random.random()
            all_zeroes = np.all(self.q[state][:] == self.q[state][0])
            if selection <= EPSILON or all_zeroes:
                return random.randrange(self.num_actions)
            else:
                return np.argmax(self.q[state][:])

    # I cannot come up with something to print
    def report(self,final):
        """
        Function to print useful information, printed during the main loop
        """
        print("Last episode duration: ", self.episode_durrations[-1])
        print("Average Epiusode duration: ", sum(self.episode_durrations) / len(self.episode_durrations))
        if len(self.episode_durrations) > 20:
            print("Rolling average duration of last 20 episodes: ", sum(self.episode_durrations[-20:])/20)
        
        if final:
            return self.episode_durrations
        print("---")

    def value_iteration(self, env):
        Q = np.zeros((self.num_states, self.num_actions))
        while True:
            max_D = 0
            prev_Q = Q.copy()                                   # copy of the current Q values
            for s in range(self.num_states-1):                  # terminal state is ignored, obviously
                for a in range(self.num_actions):               # P dictionary dict of dicts of lists, where
                    transitions = env.P[s][a]                   # P[s][a] == [(probability, nextstate, reward, done), ...]
                    sum = 0
                    for t in transitions:
                        prob_t = t[0]                           # probability of transitioning to s'
                        next_state = t[1]                       # s'
                        r = t[2]                                # R(s, a, s')
                        prevQs = np.max(prev_Q[next_state][:])  # max of previous Q values for s'
                        sum += prob_t * (r + self.discount * prevQs)
                    Q[s][a] = sum
                    if max_D < abs((Q[s][a] - prev_Q[s][a])):
                        max_D = abs((Q[s][a] - prev_Q[s][a]))
            if max_D < VI_THRESHOLD:
                break
        return Q

    def extract_policy(self, Q):
        policy = []
        for s in range(self.num_states):
            policy.append(np.argmax(Q[s][:]))
        return policy
