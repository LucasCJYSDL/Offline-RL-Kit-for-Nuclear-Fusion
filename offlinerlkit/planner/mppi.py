import numpy as np
from tqdm import tqdm
from copy import deepcopy

from envs.env_wrappers import PlanningWrapper
from offlinerlkit.utils.logger import Logger

class MPPI:
    def __init__(self,
                 plan_env: PlanningWrapper,
                 args,
                 logger: Logger,
                 device,
                 eval_env: PlanningWrapper = None,
                 uncertainty_mode = "ensemble_std"
                 ):
        self.plan_env = plan_env
        self.eval_env = eval_env if eval_env is not None else plan_env
        self.use_separate_eval_env = eval_env is not None
        self.obs_dim, self.action_dim = args.obs_shape[0], args.action_dim
        self.max_action = args.max_action
        self.min_action = args.min_action

        self.logger = logger
        self.device = device

        # core parameters
        self.gamma = args.gamma
        self.episodes_per_shot = args.episodes_per_shot
        self.num_envs = args.num_envs
        self.horizon = args.horizon
        self.num_samples = args.num_samples
        self.lam = args.lam
        self.penalty_coef = args.penalty_coef
        self.uncertainty_mode = uncertainty_mode
        self.planner_seed = getattr(args, "planner_seed", getattr(args, "seed", None))
        self.rng = np.random.default_rng(self.planner_seed)

        assert self.num_samples % self.num_envs == 0
        self.num_batches = self.num_samples // self.num_envs # in case the num of samples is too large

    def run(self):
        rollouts, dataset = self.run_rollouts()
        return dataset

    def run_rollouts(self):
        shot_list = self.eval_env.get_reference_shots()
        # we can collect the planning results into a dataset in d4rl format
        dataset = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'terminals': []}
        rollouts = {}

        for ref_shot in shot_list:
            print("Running MPPI for Shot #{}".format(ref_shot))
            # for each shot, we repeat the algorithm for several times
            env_return_list = []
            rollouts[int(ref_shot)] = []
            for exp_id in range(self.episodes_per_shot):
                rollout = self._run_episode(ref_shot)
                rollouts[int(ref_shot)].append(rollout)
                for key in dataset:
                    dataset[key].extend(rollout[key])
                if dataset['terminals']:
                    dataset['terminals'][-1] = True
                    rollout['terminals'][-1] = True
                env_return = float(np.sum(rollout['rewards']))
                env_return_list.append(env_return)
                print("Trial # {} with return {}".format(exp_id, env_return))

            # end of planning for a certain shot
            self.logger.set_timestep(ref_shot)
            self.logger.logkv("return_mean", np.mean(env_return_list))
            self.logger.logkv("return_std", np.std(env_return_list))
            self.logger.dumpkvs(exclude=["policy_training_progress", "tb"])

        for k in dataset:
            dataset[k] = np.array(dataset[k])

        for shot_rollouts in rollouts.values():
            for rollout in shot_rollouts:
                for k in rollout:
                    rollout[k] = np.array(rollout[k])

        return rollouts, dataset

    def _run_episode(self, ref_shot):
        # reset the planning model and the evaluation model at the same shot
        if self.use_separate_eval_env:
            self.plan_env.reset(ref_shot)
            env_state = self.eval_env.reset(ref_shot)
            self.plan_env.sync_runtime_state_from(self.eval_env)
            total_steps = self.eval_env.get_shot_length()
        else:
            env_state = self.plan_env.reset(ref_shot)
            total_steps = self.plan_env.get_shot_length()
        env_state = env_state[0].cpu().numpy() # a little bit hacky
        horizon = min(self.horizon, total_steps)

        # Initialize control sequence
        u_init = np.zeros((total_steps, self.action_dim)) # TODO: use a policy function to provide the initial action sequence
        u = u_init.copy()[:horizon]
        # noise_covariance = np.array([np.eye(self.action_dim) for _ in range(horizon)]) # TODO

        rollout = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': [],
            'terminals': [],
            'time_steps': [],
        }

        for t_step in tqdm(range(total_steps)):
            rollout['observations'].append(env_state.copy())

            # sample actions
            noise = self.rng.normal(loc=0.0, scale=1.0, size=(self.num_samples, horizon, self.action_dim))
            # # TODO: generate noise based on the covirance, which does not assume independence
            # noise = []
            # for h_idx in range(horizon):
            #     noise.append(np.random.multivariate_normal(mean=np.zeros(self.action_dim), cov=noise_covariance[h_idx], size=self.num_samples))
            # noise = np.stack(noise, axis=1)
            actions = u[None, :, :] + noise  # Shape: (num_samples, horizon, action_dim)

            # clip actions to the action space and get the clipped noise
            actions = np.clip(actions, self.min_action, self.max_action)
            noise = actions - u[None, :, :]

            # lookahead
            total_cost = np.zeros(self.num_samples, dtype=np.float32)
            # dones = np.zeros(self.num_samples, dtype=np.float32)
            # final_states = np.zeros((self.num_samples, self.obs_dim), dtype=np.float32)
            for batch_idx in range(self.num_batches):
                # get a shot of the current env to start the planning process
                temp_env = deepcopy(self.plan_env)
                mask = np.ones(self.num_envs, dtype=bool)
                s_id, e_id = batch_idx * self.num_envs, (batch_idx+1) * self.num_envs
                batch_actions = actions[s_id: e_id]

                for h in range(horizon):
                    a = batch_actions[:, h, :]
                    s, r, d, info = temp_env.step(a) # (num_envs, state_dim), (num_envs, ), (num_envs, )
                    s = s.cpu().numpy()
                    r = np.asarray(r).reshape(-1)
                    d = np.asarray(d).reshape(-1)

                    # different from usual MPPI
                    if self.penalty_coef:
                        r -= self.penalty_coef * self._get_reward_penalty(info["means"], info["stds"])

                    total_cost[s_id: e_id][mask] -= (self.gamma ** h) * r[mask]
                    # final_states[s_id: e_id][mask] = s[mask]
                    # dones[s_id: e_id][mask] = d[mask]

                    # once an env terminates, it would be masked util the end of the horizon
                    mask = np.logical_and(mask, np.logical_not(d))
                    if not mask.any():
                        break
            # TODO: use a trained value function to provide truncated values based on the final states

            # update the action sequence based on the lookahead results
            beta = np.min(total_cost)
            weights = np.exp(-1 / self.lam * (total_cost - beta))
            weights /= np.sum(weights)

            weighted_noise = np.einsum('i,ijk->jk', weights, noise) # (1000,) (1000, 20, 6) (20, 6)
            u += weighted_noise
            u = np.clip(u, self.min_action, self.max_action)

            ## danger: update the noise covariance based on the lookahead results
            # covariance = np.einsum('...i,...j->...ij', noise, noise) # (1000, 20, 6, 6)
            # weighted_covariance = np.tensordot(weights, covariance, axes=1) # (20, 6, 6)

            # execute the first action in the control sequence
            action_to_take = u[0:1]
            if self.use_separate_eval_env:
                self.plan_env.step(action_to_take)
                env_state, env_reward, env_done, info = self.eval_env.step(action_to_take)
                self.plan_env.sync_runtime_state_from(self.eval_env)
            else:
                env_state, env_reward, env_done, info = self.plan_env.step(action_to_take)
            env_state = env_state.cpu().numpy()[0]
            rollout['actions'].append(action_to_take[0])
            rollout['rewards'].append(env_reward)
            rollout['terminals'].append(env_done)
            rollout['next_observations'].append(env_state.copy())
            rollout['time_steps'].append(info["time_step"] - 1)
            if env_done:
                break

            # shift the control sequence
            u = np.roll(u, -1, axis=0)
            if t_step + horizon < len(u_init):
                u[-1] = u_init[t_step + horizon]
            else:
                u[-1] = 0.0  # danger, but these extra actions won't influence the lookahead results

            ## danger: shift the noise covariance sequence
            # noise_covariance = np.roll(weighted_covariance, -1, 0)
            # noise_covariance[-1] = np.eye(action_dim)

        return rollout

    def _get_reward_penalty(self, mean, std):
        """
        Compute reward penalty based on the ensemble predictions. Copied from MOPO.
        """
        if self.uncertainty_mode == "aleatoric":
            penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
        elif self.uncertainty_mode == "pairwise-diff":
            next_obses_mean = mean
            next_obs_mean = np.mean(next_obses_mean, axis=0)
            diff = next_obses_mean - next_obs_mean
            penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
        elif self.uncertainty_mode == "ensemble_std":
            next_obses_mean = mean
            penalty = np.sqrt(next_obses_mean.var(0).mean(1))
        else:
            raise ValueError

        return penalty
