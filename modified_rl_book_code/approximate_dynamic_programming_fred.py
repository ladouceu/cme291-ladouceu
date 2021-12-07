'''Approximate dynamic programming algorithms are variations on
dynamic programming algorithms that can work with function
approximations rather than exact representations of the process's
state space.

'''

from typing import Iterator, Tuple, TypeVar, Sequence, List
from operator import itemgetter
import numpy as np

from rl.distribution import Distribution
from .function_approx_fred import FunctionApprox
from rl.iterate import iterate
from rl.markov_process import (FiniteMarkovRewardProcess, MarkovRewardProcess,
                               RewardTransition, NonTerminal, State)
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        MarkovDecisionProcess,
                                        StateActionMapping)
from rl.policy import DeterministicPolicy


import sys
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d




S = TypeVar('S')
A = TypeVar('A')

# A representation of a value function for a finite MDP with states of
# type S
ValueFunctionApprox = FunctionApprox[NonTerminal[S]]
QValueFunctionApprox = FunctionApprox[Tuple[NonTerminal[S], A]]
NTStateDistribution = Distribution[NonTerminal[S]]


def extended_vf(vf: ValueFunctionApprox[S], s: State[S]) -> float:
    return s.on_non_terminal(vf, 0.0)


def evaluate_finite_mrp(
        mrp: FiniteMarkovRewardProcess[S],
        γ: float,
        approx_0: ValueFunctionApprox[S]
) -> Iterator[ValueFunctionApprox[S]]:

    '''Iteratively calculate the value function for the give finite Markov
    Reward Process, using the given FunctionApprox to approximate the
    value function at each step.

    '''
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:
        vs: np.ndarray = v.evaluate(mrp.non_terminal_states)
        updated: np.ndarray = mrp.reward_function_vec + γ * \
            mrp.get_transition_matrix().dot(vs)
        return v.update(zip(mrp.non_terminal_states, updated))

    return iterate(update, approx_0)


def evaluate_mrp(
    mrp: MarkovRewardProcess[S],
    γ: float,
    approx_0: ValueFunctionApprox[S],
    non_terminal_states_distribution: NTStateDistribution[S],
    num_state_samples: int
) -> Iterator[ValueFunctionApprox[S]]:

    '''Iteratively calculate the value function for the given Markov Reward
    Process, using the given FunctionApprox to approximate the value function
    at each step for a random sample of the process' non-terminal states.

    '''
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:
        nt_states: Sequence[NonTerminal[S]] = \
            non_terminal_states_distribution.sample_n(num_state_samples)

        def return_(s_r: Tuple[State[S], float]) -> float:
            s1, r = s_r
            return r + γ * extended_vf(v, s1)

        return v.update(
            [(s, mrp.transition_reward(s).expectation(return_))
             for s in nt_states]
        )

    return iterate(update, approx_0)


def value_iteration_finite(
    mdp: FiniteMarkovDecisionProcess[S, A],
    γ: float,
    approx_0: ValueFunctionApprox[S]
) -> Iterator[ValueFunctionApprox[S]]:
    '''Iteratively calculate the Optimal Value function for the given finite
    Markov Decision Process, using the given FunctionApprox to approximate the
    Optimal Value function at each step

    '''
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:

        def return_(s_r: Tuple[State[S], float]) -> float:
            s1, r = s_r
            return r + γ * extended_vf(v, s1)

        return v.update(
            [(
                s,
                max(mdp.mapping[s][a].expectation(return_)
                    for a in mdp.actions(s))
            ) for s in mdp.non_terminal_states]
        )

    return iterate(update, approx_0)


def value_iteration(
    mdp: MarkovDecisionProcess[S, A],
    γ: float,
    approx_0: ValueFunctionApprox[S],
    non_terminal_states_distribution: NTStateDistribution[S],
    num_state_samples: int
) -> Iterator[ValueFunctionApprox[S]]:
    '''Iteratively calculate the Optimal Value function for the given
    Markov Decision Process, using the given FunctionApprox to approximate the
    Optimal Value function at each step for a random sample of the process'
    non-terminal states.

    '''
    def update(v: ValueFunctionApprox[S]) -> ValueFunctionApprox[S]:
        nt_states: Sequence[NonTerminal[S]] = \
            non_terminal_states_distribution.sample_n(num_state_samples)

        def return_(s_r: Tuple[State[S], float]) -> float:
            s1, r = s_r
            return r + γ * extended_vf(v, s1)

        return v.update(
            [(s, max(mdp.step(s, a).expectation(return_)
                     for a in mdp.actions(s)))
             for s in nt_states]
        )

    return iterate(update, approx_0)


def backward_evaluate_finite(
    step_f0_pairs: Sequence[Tuple[RewardTransition[S],
                                  ValueFunctionApprox[S]]],
    γ: float
) -> Iterator[ValueFunctionApprox[S]]:
    '''Evaluate the given finite Markov Reward Process using backwards
    induction, given that the process stops after limit time steps.

    '''

    v: List[ValueFunctionApprox[S]] = []

    for i, (step, approx0) in enumerate(reversed(step_f0_pairs)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            return r + γ * (extended_vf(v[i-1], s1) if i > 0 else 0.)

        v.append(
            approx0.solve([(s, res.expectation(return_))
                           for s, res in step.items()])
        )

    return reversed(v)


MRP_FuncApprox_Distribution = Tuple[MarkovRewardProcess[S],
                                    ValueFunctionApprox[S],
                                    NTStateDistribution[S]]


def backward_evaluate(
    mrp_f0_mu_triples: Sequence[MRP_FuncApprox_Distribution[S]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[ValueFunctionApprox[S]]:
    '''Evaluate the given finite Markov Reward Process using backwards
    induction, given that the process stops after limit time steps, using
    the given FunctionApprox for each time step for a random sample of the
    time step's states.

    '''
    v: List[ValueFunctionApprox[S]] = []

    for i, (mrp, approx0, mu) in enumerate(reversed(mrp_f0_mu_triples)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            return r + γ * (extended_vf(v[i-1], s1) if i > 0 else 0.)

        v.append(
            approx0.solve(
                [(s, mrp.transition_reward(s).expectation(return_))
                 for s in mu.sample_n(num_state_samples)],
                error_tolerance
            )
        )

    return reversed(v)


def back_opt_vf_and_policy_finite(
    step_f0s: Sequence[Tuple[StateActionMapping[S, A],
                             ValueFunctionApprox[S]]],
    γ: float,
) -> Iterator[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]]:
    '''Use backwards induction to find the optimal value function and optimal
    policy at each time step

    '''
    vp: List[Tuple[ValueFunctionApprox[S], DeterministicPolicy[S, A]]] = []

    for i, (step, approx0) in enumerate(reversed(step_f0s)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            return r + γ * (extended_vf(vp[i-1][0], s1) if i > 0 else 0.)

        this_v = approx0.solve(
            [(s, max(res.expectation(return_)
                     for a, res in actions_map.items()))
             for s, actions_map in step.items()]
        )

        def deter_policy(state: S) -> A:
            return max(
                ((res.expectation(return_), a) for a, res in
                 step[NonTerminal(state)].items()),
                key=itemgetter(0)
            )[1]

        vp.append((this_v, DeterministicPolicy(deter_policy)))

    return reversed(vp)


MDP_FuncApproxV_Distribution = Tuple[
    MarkovDecisionProcess[S, A],
    ValueFunctionApprox[S],
    NTStateDistribution[S]
]


# source: https://scipy.github.io/old-wiki/pages/Cookbook/SavitzkyGolay
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
#     except ValueError, msg:
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')




def back_opt_vf_and_policy(
    mdp_f0_mu_triples: Sequence[MDP_FuncApproxV_Distribution[S, A]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float,
    mode:str
) -> Iterator[Tuple[ValueFunctionApprox[S],
                    ValueFunctionApprox[S],
                    ValueFunctionApprox[S],
                    DeterministicPolicy[S, A]]]:
    '''Use backwards induction to find the optimal value function and optimal
    policy at each time step, using the given FunctionApprox for each time step
    for a random sample of the time step's states.

    '''
    vp: List[Tuple[ValueFunctionApprox[S],
                   ValueFunctionApprox[S],
                   ValueFunctionApprox[S],
                   DeterministicPolicy[S, A]
                   ]] = []

    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):
        
        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            return r + γ * (extended_vf(vp[i-1][0], s1) if i > 0 else 0.)
        
        if mode == "call":
            data_batch_optimal = [(s, max(mdp.step(s, a).expectation(return_)
                                          for a in mdp.actions(s)))
                                 for s in mu.sample_n(num_state_samples)]
            data_batch_continue = [(s, mdp.step(s, False).expectation(return_))
                                  for s in mu.sample_n(num_state_samples)]
            data_batch_execute  = [(s, mdp.step(s, True).expectation(return_))
                                  for s in mu.sample_n(num_state_samples)]
        
        # sample from uniform distribution instead of from mu, I am having trouble sampling outliers and 
        # I know that the derivative prices that are intersting are only between 0 and the spot price.
        if mode == "put":
            data_batch_optimal = [(NonTerminal(s), max(mdp.step(NonTerminal(s), a).expectation(return_)
                                          for a in mdp.actions(NonTerminal(s))))
                                 for s in (250*np.random.rand(num_state_samples)).tolist()]
            data_batch_continue = [(NonTerminal(s), mdp.step(NonTerminal(s), False).expectation(return_))
                                  for s in (250*np.random.rand(num_state_samples)).tolist()]
            data_batch_execute  = [(NonTerminal(s), mdp.step(NonTerminal(s), True).expectation(return_))
                                  for s in (250*np.random.rand(num_state_samples)).tolist()]

#             data_batch_optimal = [(NonTerminal(s), max(mdp.step(NonTerminal(s), a).expectation(return_)
#                                           for a in mdp.actions(NonTerminal(s))))
#                                  for s in mu.sample_n(num_state_samples)]
#             data_batch_continue = [(NonTerminal(s), mdp.step(NonTerminal(s), False).expectation(return_))
#                                   for s in mu.sample_n(num_state_samples)]
#             data_batch_execute  = [(NonTerminal(s), mdp.step(NonTerminal(s), True).expectation(return_))
#                                   for s in mu.sample_n(num_state_samples)]

        
        
        # here there is no need to model all 3, only need to model 1: the continuation vf
        # we know the execution vf = payoff
        # we know the optimal: f(x) = max(payoff(x),vf_opt(x))
        # BUT I am not able to get around the library interface to build this
        v_opt  = approx0.solve(data_batch_optimal,  error_tolerance)
        v_cont = approx0.solve(data_batch_continue, error_tolerance)
        v_exec = approx0.solve(data_batch_execute,  error_tolerance)


        

        
        
#         plt.figure()
        
#         s = [s for s,_ in data_batch]
#         x = [s.state for s,_ in data_batch]
#         y = [y for _,y in data_batch]
        
#         # plot the data samples used to approximate the optimal vf
#         plt.scatter(x,y, color = "blue", s = 5)
        
#         s = [s for s,_ in data_batch_continue]
#         x = [s.state for s,_ in data_batch_continue]
#         y = [y for _,y in data_batch_continue]
        
#         # plot the data samples used to approximate the optimal vf
#         plt.scatter(x,y, color = "green", s = 5)
        
#         s = [s for s,_ in data_batch_execute]
#         x = [s.state for s,_ in data_batch_execute]
#         y = [y for _,y in data_batch_execute]
        
#         # plot the data samples used to approximate the optimal vf
#         plt.scatter(x,y, color = "red", s = 5)
        
# #         # plot the ADP curve
        
# #         y_pred = this_v.evaluate(s)
# #         plt.scatter(x, y_pred, color="r", s = 5)
        
# # #         # plot the sg filtering curve
# #         x_array = np.array(x)
# #         y_array = np.array(y)
# #         sorted_indices = np.argsort(x_array)
# #         x_sorted = x_array[sorted_indices]
# #         y_sorted = y_array[sorted_indices]
# # #         y_sg = savitzky_golay(y_sorted, 11, 2)
# # # #         plt.plot(x_sorted, y_sg, color = "green")
        
# #         # try with linear interpolation
# #         interp = interp1d(x_sorted, y_sorted, fill_value = "extrapolate")
# #         x_interp = np.arange(x_array.min()*0.7, x_array.max()*1.3, 1)
# #         plt.plot(x_interp, interp(x_interp), color = "green")
        

#         plt.show()
        
        
        # here there is no need to do this computation again, I can simply use the model of the vf_cont and payoff function to find
        # the maximum price x at which payoff(x) >= vf_cont(x) holds.
        # a small tolerance could be useful here to account for approximation errors; the vf_cont can get very close to the payoff 
        # without crossing it
#         def deter_policy(state: S) -> A:
#             return max(
#                 ((mdp.step(NonTerminal(state), a).expectation(return_), a)
#                  for a in mdp.actions(NonTerminal(state))),
#                 key=itemgetter(0)
#             )[1]

#         vp.append((v_opt,
#                    v_cont,
#                    v_exec,
#                    DeterministicPolicy(deter_policy)))
        def deter_policy(state: S) -> A:
            return max(((v_cont(NonTerminal(state)), False),(v_exec(NonTerminal(state)), True)), 
                       key=itemgetter(0)
                      )[1]

        vp.append((v_opt,
                   v_cont,
                   v_exec,
                   DeterministicPolicy(deter_policy)))
        

    return reversed(vp)


MDP_FuncApproxQ_Distribution = Tuple[
    MarkovDecisionProcess[S, A],
    QValueFunctionApprox[S, A],
    NTStateDistribution[S]
]


def back_opt_qvf(
    mdp_f0_mu_triples: Sequence[MDP_FuncApproxQ_Distribution[S, A]],
    γ: float,
    num_state_samples: int,
    error_tolerance: float
) -> Iterator[QValueFunctionApprox[S, A]]:
    '''Use backwards induction to find the optimal q-value function  policy at
    each time step, using the given FunctionApprox (for Q-Value) for each time
    step for a random sample of the time step's states.

    '''
    horizon: int = len(mdp_f0_mu_triples)
    qvf: List[QValueFunctionApprox[S, A]] = []

    for i, (mdp, approx0, mu) in enumerate(reversed(mdp_f0_mu_triples)):

        def return_(s_r: Tuple[State[S], float], i=i) -> float:
            s1, r = s_r
            next_return: float = max(
                qvf[i-1]((s1, a)) for a in
                mdp_f0_mu_triples[horizon - i][0].actions(s1)
            ) if i > 0 and isinstance(s1, NonTerminal) else 0.
            return r + γ * next_return

        this_qvf = approx0.solve(
            [((s, a), mdp.step(s, a).expectation(return_))
             for s in mu.sample_n(num_state_samples) for a in mdp.actions(s)],
            error_tolerance
        )

        qvf.append(this_qvf)

    return reversed(qvf)
