from typing import Optional
import numpy as np
from numpy.typing import NDArray
from enum import Enum
import random
import matplotlib.pyplot as plt


"""
| (0,0) | (0, 1) | (0, 2) | (0, 3) | 
| (1,0) | (1, 1) | (1, 2) | (1, 3) | 
| (2,0) | (2, 1) | (2, 2) | (2, 3) | 
| (3,0) | (3, 1) | (3, 2) | (3, 3) | 

"""

class Action(Enum):
    DOWN = (1, 0)
    UP = (-1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)

    @classmethod
    def all(cls):
        return [cls.RIGHT, cls.LEFT, cls.UP, cls.DOWN]

    @classmethod
    def getIndex(cls, action: 'Action') -> int:
        return Action.all().index(action)



class GridWorld:
    def __init__(self, states_features: Optional[NDArray] = None, size = 5):
        self.size = size
        self.num_states = size * size 
        self.initial_state = (0, 0)
        self.terminal_state = (size-1, size-1)
        self.states = [(i, j) for i in range(size) for j in range(size)]

        self.ice_prob = 0.2 # Probability of not moving to the intended state

        
        if states_features is None:
            self.states_features = np.identity(self.num_states)
        else:
            self.states_features = states_features

    def stateIndex(self, state: tuple[int, int]):
        return self.states.index(state)
    
    def adjacent_states(self, state: tuple[int, int]) -> list[tuple[int, int]]:
        """Return the states reachable in one step from `state`."""
        adjacent = []
        v, h = state
        if v > 0:
                adjacent.append((v-1, h))
        if h > 0:
            adjacent.append((v, h-1))
        if v < self.size-1:
            adjacent.append((v+1, h))
        if h < self.size - 1:
            adjacent.append((v, h+1))
        return adjacent


    def get_next_deterministic(self, state: tuple[int, int], action: Action) -> tuple[int, int]:
        (v, h) = state
        (va, ha) = action.value 
        nxt_state = v + va, h + ha 

        if nxt_state not in self.states:
            return state 
        return nxt_state


    def get_next_state(self, state: tuple[int, int], action: Action) -> tuple[int,int]:
        """
        Apply the action to the given state, to find the next state.
        """
        if random.random() < self.ice_prob:
            # Go to one of adjacent states, at random
            av_states = self.adjacent_states(state)
            return random.choice(av_states)
        return self.get_next_deterministic(state, action)
    
    def get_next_state_probability(self, state: tuple[int, int], action: Action, nxt_state: tuple[int, int]) -> float:
        """
        Get the probability of going from `state` to `nxt_state` when executing `action`.
        """
        possible_next_states = self.adjacent_states(state)
        next_deterministic = self.get_next_deterministic(state, action)

        # Staying on the same state: can only happen deterministically
        if nxt_state == state:
            if next_deterministic == nxt_state:
                return 1 - self.ice_prob
            return 0.
        
        # Moving to a different state
        if next_deterministic == nxt_state:
            return (1-self.ice_prob)  + self.ice_prob / len(possible_next_states)
        
        if nxt_state  in possible_next_states:
            return  self.ice_prob / len(possible_next_states)
        
        return 0.

    
def get_expert_action(stateIdx: int, expert_policy: NDArray) -> Action: 
    """
    Get the action that the expert would take, from a nondeterministic policy
    """
    pol = expert_policy[stateIdx]
    prob = random.random()
    for i, p in enumerate(pol):
        prob -= p 
        if prob <= 0:
            return Action.all()[i]
    return Action.all()[-1]

def get_expert_trajectory(world: GridWorld, expert_policy: NDArray) -> list[tuple[int, int]]:
    """Get a trajectory from the expert"""
    traj: list[tuple[int, int]] = [world.initial_state]
    while traj[-1] != world.terminal_state:
        # Repeat until terminal state not reached
        act = get_expert_action(world.stateIndex(traj[-1]), expert_policy)
        traj.append(world.get_next_state(traj[-1], act))
    return traj


def compute_feature_avg(num_traj: int, world: GridWorld, expert_policy: NDArray) -> NDArray:
    """
    Compute the expected number of features per trajectory.
    It calculates the average over a set of expert trajectories.
    """
    trajectories = [get_expert_trajectory(world, expert_policy) for _ in range(num_traj)]

    avg_features = np.zeros(len(world.states_features[0, :]))
    for traj in trajectories:
        for state in traj:
            avg_features += world.states_features[world.stateIndex(state)]

    return avg_features / num_traj

def get_value(world: GridWorld, rew_weights: NDArray, eps=1e-4, gamma=0.9) -> NDArray:
    """Compute the value function of the optimal policy, for the given reward weights"""
    p = [np.matrix([[world.get_next_state_probability(state, a, nxt_state) for nxt_state in world.states] for state in world.states]) for a in Action.all()]
    v = np.zeros(world.num_states)
    delta = np.inf
    while delta > eps:   
        v_old = v
        q = gamma * np.array([p[a] @ v for a in range(4)])

        v = world.states_features.dot(rew_weights) + np.max(q, axis=0)[0]

        delta = np.max(np.abs(v_old - v))
    return v
    
def generate_policy(world: GridWorld, rew_weights: NDArray) -> NDArray:
    """Compute a stochastic policy given this reward weights"""
    value = np.exp(get_value(world, rew_weights))     

    q = np.array([
        np.array([value[world.stateIndex(world.get_next_deterministic(s, a))] for a in Action.all()])
        for s in world.states
    ])

    return q / np.sum(q, axis=1)[:, None]


def compute_frequency(world: GridWorld, weights: NDArray, N = 10):
    """Compute the visitation frequency of all states, given this reward weights"""
    Z_s = [0. for _ in range(world.num_states)]
    Z_s[-1] = 1.
    Z_sa = [[0., 0., 0., 0.] for _ in range(world.num_states)]

    # Backward pass
    for _ in range(N):
        for state in world.states:
            for action in Action.all():
                Z_sa[world.stateIndex(state)][Action.getIndex(action)] = 0.
                for nxt_state in world.states:
                    Z_sa[world.stateIndex(state)][Action.getIndex(action)] += world.get_next_state_probability(state, action, nxt_state) * np.exp(weights.dot(world.states_features[world.stateIndex(state)])) * Z_s[world.stateIndex(nxt_state)] # type: ignore
        for state in world.states:
            Z_s[world.stateIndex(state)] = sum(Z_sa[world.stateIndex(state)])

    # Local probabilities
    local_probs = []
    for state in world.states:
        p = []
        for act in Action.all():
            if Z_s[world.stateIndex(state)] == 0:
                p.append(0)
            else:
                p.append(Z_sa[world.stateIndex(state)][Action.getIndex(act)] / Z_s[world.stateIndex(state)])
        local_probs.append(p)

    # forward pass
    D_ts = [[1] + [ 0 for _ in range(world.num_states - 1)]]
    for _ in range(1, N):           
        curr = []
        for nxt_state in world.states:
            tmp = 0
            for action in Action.all():
                for state in world.states:
                    tmp += D_ts[-1][world.stateIndex(state)] * world.get_next_state_probability(state, action, nxt_state) * local_probs[world.stateIndex(state)][Action.getIndex(action)]
            curr.append(tmp)
        D_ts.append(curr)

    D_s = np.zeros(world.num_states)
    for ds in D_ts:
        D_s += np.asarray(ds)

    return D_s

def maxEnt(world: GridWorld, expert_policy: NDArray, lr = 0.01, eps=1e-4, num_traj = 200, maxIterations = 500):
    """Execute the maxEnt agorithm"""
    curr_weights = np.ones(len(world.states_features[0]))

    feat_avg = compute_feature_avg(num_traj, world, expert_policy)

    def get_gradient():
        """Compute the current gradient"""
        frequency = compute_frequency(world, curr_weights, N = 50)
        gradient = feat_avg.copy()
        for state in world.states:
            gradient -= frequency[world.stateIndex(state)] * world.states_features[world.stateIndex(state), :]
        return gradient
    
    initial_lr = lr
    delta = np.inf 
    i = 0
    while delta > eps and  i < maxIterations:
        i += 1
        if i % 50 == 0:

            print(i, delta)
        gradient = get_gradient()
        old_weights = curr_weights.copy()
        lr = initial_lr/(1.0 + i)
        curr_weights *= np.exp(lr * gradient)
        delta = np.max(np.abs(old_weights - curr_weights))
    print("Num iterations:", i)
    print("Final delta:", delta )
    return curr_weights

def print_policy(policy: NDArray, world: GridWorld, name: str):
    length = 5 * world.size + 1
    print("-" * ((length- len(name))//2 -1)  + f" {name} " + "-" * ((length - len(name))//2 -1))    
    for i in range(world.size):
        st = "| "
        for j in range(world.size):
            act_idx = np.argmax(policy[world.size*i+j])
            action = Action.all()[act_idx]
            if action == Action.RIGHT:
                st += "-> | "
            elif action == Action.LEFT:
                st += "<- | "
            elif action == Action.DOWN:
                st += "\\/ | "
            elif action == Action.UP:
                st += "/\\ | "
        print(st)
    print( "-" * length)


def plot_rewards(real_weights: NDArray, my_weights: NDArray, world: GridWorld):
    my_rewards = world.states_features @ my_weights
    table = np.asarray([my_rewards[:5],
        my_rewards[5:10],
        my_rewards[10:15],
        my_rewards[15:20],
        my_rewards[20:25]])

    real_rewards = world.states_features @ real_weights
    original_table = np.asarray([real_rewards[:5],
        real_rewards[5:10],
        real_rewards[10:15],
        real_rewards[15:20],
        real_rewards[20:25]])


    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('Original Reward') #type:ignore
    _ = ax1.imshow(original_table)  
    ax2 = fig.add_subplot(122)
    ax2.title.set_text('Computed Reward') #type: ignore
    _ = ax2.imshow(table)
    plt.show()



if __name__ == "__main__":
    # Create the grid world and the weights

    # ---- Using a separate feature for each state ----
    #world = GridWorld(size = 5, states_features=np.identity(25))

    #real_weights = np.zeros(world.num_states)
    #real_weights[world.stateIndex(world.terminal_state)] = 1.
    #real_weights[world.stateIndex((1, 1))] = 0.65
    #real_weights[world.stateIndex((2, 3))] = 0.8

    # ---- Using same feature for states with same reward, different otherwise ----
    #world = GridWorld(size=5, states_features=np.array([np.array([1, 0, 0, 0]) for _ in range(25)]))
    #world.states_features[world.stateIndex(world.terminal_state)] = np.array([0, 0, 0, 1])
    #world.states_features[world.stateIndex((1, 1))] = np.array([0, 0, 1, 0])
    #world.states_features[world.stateIndex((2, 3))] = np.array([0, 1, 0, 0])

    #real_weights = np.array([0, 0.8, 0.65, 1])

    # ---- Using overlapping features ----
    world = GridWorld(size=5, states_features=np.array([np.array([1, 0, 0]) for _ in range(25)]))
    world.states_features[world.stateIndex(world.terminal_state)] = np.array([0, 1, 2])
    world.states_features[world.stateIndex((1, 1))] = np.array([0, 1, 0])
    world.states_features[world.stateIndex((2, 3))] = np.array([0, 1, 1])

    real_weights = np.array([0, 0.6, 0.2])


    # Generate the policy of the expert
    expert_policy = generate_policy(world, real_weights)

    # Run the MaxEnt algorithm
    my_weights = maxEnt(world, expert_policy)


    my_policy = generate_policy(world, my_weights)
    print_policy(expert_policy, world, "Expert policy")
    print_policy(my_policy, world ,"Learned policy")
    print("Computed weights:", my_weights)
    print("Real weights:", real_weights)

    plot_rewards(real_weights, my_weights, world)


