import logging
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import numpy as np
import sys


class Bandit(ABC):
    """
    Abstract Base Class defining the structure for bandit algorithms. 
    Abstract Methods:
        - __init__: Initializes the bandit's parameters.
        - __repr__: String representation of the bandit.
        - update: Updates the bandit's parameters according to reward.
        - report: Reports results and saves them to CSV files.
        - pull: Simulates pulling the bandit's arm for reward.
        - experiment: Runs the bandit experiment.

    """
    @abstractmethod
    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0
        self.r_estimate = 0

    @abstractmethod
    def __repr__(self):
        return f'Arm with Win rate is {self.p}'

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self, results, algorithm, N):
        """
        Reports results then saves data to CSV and logs insights.
        """
        if algorithm == 'EpsilonGreedy':
            (cumul_reward_avg, cumul_reward, cumul_regret,
             bandits, chosen_bandit, reward, count_suboptimal) = results
        else:
            (cumul_reward_avg, cumul_reward, cumul_regret,
             bandits, chosen_bandit, reward) = results

        # Save data to CSV
        experiment = pd.DataFrame({
            'Trial': range(N),
            'Bandit': chosen_bandit,
            'Reward': reward,
            'Algorithm': algorithm
        })
        experiment.to_csv(f'{algorithm}_Experiment.csv', index=False)

        # Save final results to CSV
        final_results = pd.DataFrame({
            'Bandit': [b.p for b in bandits],
            'Estimated_Reward': [b.p_estimate for b in bandits],
            'Pulls': [b.N for b in bandits],
            'Estimated_Regret': [b.r_estimate for b in bandits]
        })
        final_results.to_csv(f'{algorithm}_Final_Result.csv', index=False)

        # Logging
        logger.info(f"--- {algorithm} Reporting ---")
        for b in range(len(bandits)):
            logger.debug(
                f"Bandit with true Win rate {bandits[b].p:.3f}: "
                f"Pulled {bandits[b].N} times "
                f"Estimated Reward is {bandits[b].p_estimate:.4f}, "
                f"Estimated Regret is {bandits[b].r_estimate:.4f}"
            )
        logger.info(f"Cumulative Reward is {sum(reward):.2f}")
        logger.info(f"Cumulative Regret is {cumul_regret[-1]:.2f}")

        if algorithm == 'EpsilonGreedy':
            suboptimal_rate = count_suboptimal / N
            logger.warning(f"Suboptimal Pull Percentage is {suboptimal_rate:.4%}")

        # Save cumulative rewards, regrets
        summary = pd.DataFrame({
            'Trial': range(N),
            'Cumulative Reward': cumul_reward,
            'Cumulative Regret': cumul_regret
        })
        summary.to_csv(f'{algorithm}_Cumulative_Results.csv', index=False)


class Visualization:
    """
    Methods for visualizing the performance of bandit algorithms.
    Methods:
        - plot_rewards: Visualizes average reward convergence over trials for both linear and log scale.
        - plot_regrets: Visualizes average regret convergence over trials for both linear and log scale.
    """
    @classmethod
    def plot_rewards(cls, rewards, num_trials, optimal_bandit_reward, title="Average Reward Convergence"):
        """
        Plotting average reward convergence of algorithms.
        """
        cumul_rewards = np.cumsum(rewards)
        average_reward = cumul_rewards / (np.arange(num_trials) + 1)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(average_reward, label="Average Reward")
        ax[0].axhline(optimal_bandit_reward,  color="r" , label="Optimal Bandit Reward")
        ax[0].legend()
        ax[0].set_title(f"{title} Linear")
        ax[0].set_xlabel("# Trials")
        ax[0].set_ylabel("Avg Reward")

        ax[1].plot(average_reward, label="Avg Reward")
        ax[1].axhline(optimal_bandit_reward, color="r", label="Optimal Bandit Reward")
        ax[1].legend()
        ax[1].set_title(f"{title} Log")
        ax[1].set_xlabel("# Trials")
        ax[1].set_ylabel("Avg Reward")
        ax[1].set_xscale("log")

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    @classmethod
    def plot_regrets(cls, rewards, num_trials, optimal_bandit_reward,title="Average Regret Convergence"):
        """
        Plotting average regret convergence of algorithms
        """
        cumul_rewards = np.cumsum(rewards)
        cumul_regrets = optimal_bandit_reward * np.arange(1, num_trials + 1) - cumul_rewards
        average_regrets = cumul_regrets / (np.arange(num_trials) + 1)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(average_regrets, label="Avg Regret")
        ax[0].legend()
        ax[0].set_title(f"{title} Linear")
        ax[0].set_xlabel("# Trials")
        ax[0].set_ylabel("Avg Regret")
        ax[1].plot(average_regrets, label="Avg Regret")
        ax[1].legend()
        ax[1].set_title(f"{title} Log")
        ax[1].set_xlabel("# Trials")
        ax[1].set_ylabel("Avg Regret")
        ax[1].set_xscale("log")
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()


class EpsilonGreedy(Bandit):
    """
    Implementing Epsilon-Greedy algorithm

    Methods:
        - pull: Simulates pulling the arm with Gaussian noise.
        - update: Updates the bandit's estimated reward using the received reward.
        - experiment: Runs the experiment with dynamic epsilon decay.
        - report: Inherits reporting from the base Bandit class.
    """
    def __init__(self, p):
        super().__init__(p)

    def __repr__(self):
        return f"Epsilon Greedy Bandit with p = {self.p}"

    def pull(self):
        return np.random.randn() + self.p

    def update(self, x):
        self.N += 1
        self.p_estimate = (1 - 1.0 / self.N) * self.p_estimate + 1.0 / self.N * x
        self.r_estimate = self.p - self.p_estimate

    @classmethod
    def experiment(cls, BANDIT_REWARD, N, t=1):
        """
        Conducts the Epsilon-Greedy experiment
        """
        bandits = [cls(p) for p in BANDIT_REWARD]
        means = np.array(BANDIT_REWARD)
        true_best = np.argmax(means)
        count_suboptimal = 0
        EPS = 1 / t
        reward = np.empty(N)
        chosen_bandit = np.empty(N)

        for i in range(N):
            p = np.random.random()
            if p < EPS:
                j = np.random.choice(len(bandits))
            else:
                j = np.argmax([b.p_estimate for b in bandits])
            x = bandits[j].pull()
            bandits[j].update(x)
            if j != true_best:
                count_suboptimal += 1
            reward[i] = x
            chosen_bandit[i] = j
            t += 1
            EPS = 1 / t
        cumul_reward_avg = np.cumsum(reward) / (np.arange(N) + 1)
        cumul_reward = np.cumsum(reward)
        cumul_regret = (np.arange(1, N + 1) * max(means) - cumul_reward)
        return cumul_reward_avg, cumul_reward, cumul_regret, bandits, chosen_bandit, reward, count_suboptimal
    def report(self, results, algorithm, N):
        super().report(results, algorithm, N)


class ThompsonSampling(Bandit):
    """
    Implements the Thompson Sampling algorithm

    Methods:
        - pull: Simulates pulling the arm with scaled noise.
        - update: Updates posterior parameters using Bayesian update rules.
        - experiment: Runs the Thompson Sampling experiment.
        - report: Inherits reporting from the base Bandit class.
        - sample: Generates a sample from the posterior distribution.

    """
    def __init__(self, p):
        super().__init__(p)
        self.lambda_ = 1
        self.tau = 1

    def __repr__(self):
        return f"Thompson Sampling Bandit with p = {self.p}"

    def pull(self):
        return np.random.randn() / np.sqrt(self.tau) + self.p

    def sample(self):
        return np.random.randn() / np.sqrt(self.lambda_) + self.p_estimate

    def update(self, x):
        self.p_estimate = (self.tau * x + self.lambda_ * self.p_estimate) / (self.tau + self.lambda_)
        self.lambda_ += self.tau
        self.N += 1
        self.r_estimate = self.p - self.p_estimate

    @classmethod
    def experiment(cls, BANDIT_REWARD, N):
        """
        Conducts the Thompson Sampling experiment.

        """
        bandits = [cls(m) for m in BANDIT_REWARD]
        reward = np.empty(N)
        chosen_bandit = np.empty(N)

        for i in range(N):
            j = np.argmax([b.sample() for b in bandits])
            x = bandits[j].pull()
            bandits[j].update(x)

            reward[i] = x
            chosen_bandit[i] = j

        cumul_reward_avg = np.cumsum(reward) / (np.arange(N) + 1)
        cumul_reward = np.cumsum(reward)
        cumul_regret = (np.arange(1, N + 1) * max([b.p for b in bandits]) - cumul_reward)

        return cumul_reward_avg, cumul_reward, cumul_regret, bandits, chosen_bandit, reward

    def report(self, results, algorithm, N):
        super().report(results, algorithm, N)


def comparison(BANDIT_REWARD, num_trials, different_plots=False):
    """
    Compare the performance of Epsilon-Greedy and Thompson Sampling algorithms.

    Parameters:
        - bandit_rewards: List of bandit win rates.
        - num_trials: Number of trials to run.
        - different_plots: Whether to compare algorithms in separate plots.
    """
    logger.info("Comparing Epsilon Greedy vs Thompson Sampling algorithms.")
    _, _, _, _, _, eg_rewards, _ = EpsilonGreedy.experiment(BANDIT_REWARD, num_trials)
    _, _, _, _, _, ts_rewards = ThompsonSampling.experiment(BANDIT_REWARD, num_trials)

    optimal_bandit_reward = max(BANDIT_REWARD)

    Visualization.plot_rewards(eg_rewards, num_trials, optimal_bandit_reward, title="Epsilon Greedy Reward")
    Visualization.plot_rewards(ts_rewards, num_trials, optimal_bandit_reward, title="Thompson Sampling Reward")
    Visualization.plot_regrets(eg_rewards, num_trials, optimal_bandit_reward, title="Epsilon Greedy Regret")
    Visualization.plot_regrets(ts_rewards, num_trials, optimal_bandit_reward, title="Thompson Sampling Regret")

    if different_plots:
        eg_average_rewards = np.cumsum(eg_rewards) / (np.arange(num_trials) + 1)
        ts_average_rewards = np.cumsum(ts_rewards) / (np.arange(num_trials) + 1)

        plt.plot(eg_average_rewards, label="Epsilon Greedy")
        plt.plot(ts_average_rewards, label="Thompson Sampling")
        plt.axhline(optimal_bandit_reward, color="r", label="Optimal Bandit Reward")
        plt.legend()
        plt.title("Average Reward Convergence Comparison")
        plt.xlabel("Number of Trials")
        plt.ylabel("Average Reward")
        plt.xscale("log")
        plt.show()
        plt.close()


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(levelname)s : %(message)s"

    FORMATS = {logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset}

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger("BanditExperiments")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(CustomFormatter())

file_handler = logging.FileHandler("experiment.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(levelname)s : %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)


#############################################################################################################################################################


if __name__ == "__main__":
    BANDIT_REWARD = [1, 2, 3, 4]
    NUM_TRIALS = 20000

    # Run Epsilon Greedy algo
    logger.info("Conducting Epsilon-Greedy Experiment")
    results_eg = EpsilonGreedy.experiment(BANDIT_REWARD, NUM_TRIALS)  # Call directly on the class
    eg_bandit = EpsilonGreedy(0)
    eg_bandit.report(results_eg, 'EpsilonGreedy', NUM_TRIALS)

    # Run Thompson Sampling algo
    logger.info("Conducting Thompson Sampling Experiment")
    results_ts = ThompsonSampling.experiment(BANDIT_REWARD, NUM_TRIALS)  # Call directly on the class
    ts_bandit = ThompsonSampling(0)
    ts_bandit.report(results_ts, 'ThompsonSampling', NUM_TRIALS)

    # Comparing
    logger.info("Comparing of Epsilon-Greedy and Thompson Sampling")
    comparison(BANDIT_REWARD, NUM_TRIALS, different_plots=True)
    sys.exit()

