import os
import time
import numpy as np
from dexart.env.create_env import create_env
from stable_baselines3 import PPO
from examples.train import get_3d_policy_kwargs
from PIL import Image

class Evaluator:
    def __init__(self, task_name, checkpoint_path=None, use_test_set=False, headless=True):
        self.task_name = task_name
        self.checkpoint_path = checkpoint_path
        self.use_test_set = use_test_set
        self.headless = headless

        from dexart.env.task_setting import TRAIN_CONFIG, RANDOM_CONFIG
        if use_test_set:
            self.indices = TRAIN_CONFIG[task_name]['unseen']
            print(f"Using unseen instances: {self.indices}")
        else:
            self.indices = TRAIN_CONFIG[task_name]['seen']
            print(f"Using seen instances: {self.indices}")

        self.rand_pos = RANDOM_CONFIG[task_name]['rand_pos']
        self.rand_degree = RANDOM_CONFIG[task_name]['rand_degree']

        self.env = create_env(
            task_name=self.task_name,
            use_visual_obs=True,
            use_gui=True,
            is_eval=True,
            pc_noise=False,
            pc_seg=False,
            index=self.indices,
            img_type='robot',
            rand_pos=self.rand_pos,
            rand_degree=self.rand_degree
        )

        self.policy = None
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            self.policy = PPO.load(
                self.checkpoint_path,
                self.env,
                device='cuda:0',
                policy_kwargs=get_3d_policy_kwargs(extractor_name='smallpn'),
                check_obs_space=False,
                force_load=True
            )
            print("Loaded policy from checkpoint.")
        else:
            print("No valid checkpoint provided. Skipping policy loading in Evaluator.")

    def init(self):
        obs = self.env.reset()
        imgs = self.env.render(mode="rgb_array")
        if not self.headless:
            self.env.render(mode="human")
        obs.update(imgs)
        print("Environment initialized.")
        # print("obs", obs)
        return obs
        

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        imgs = self.env.render(mode="rgb_array")
        obs.update(imgs)
        """
            Observation format:
            - instance_1-seg_gt: (512, 4)
            - instance_1-point_cloud: (512, 3)
            - faucet_viz-rgb: (224, 224, 3)
            - faucet_viz2-rgb: (224, 224, 3)
            - bucket_viz-rgb: (224, 224, 3)
            - imagination_robot: (96, 7)
            - state: (32,)
            - oracle_state: (32,)
        """
        if not self.headless:
            self.env.render(mode="human")

        return obs, reward, done, info


if __name__ == "__main__":
    import argparse

    max_step = 100
    img_save_path = "./tmp"
    os.makedirs(img_save_path, exist_ok=True)
    headless = True # !![DEBUG] Known issue: if headless=False, the obs img would contain UI elements!
    save_img = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, required=True, help="Task name for the environment.")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to the trained model checkpoint.")
    parser.add_argument('--use_test_set', action='store_true', default=False, help="Use test set instances.")
    args = parser.parse_args()

    evaluator = Evaluator(
        task_name=args.task_name,
        checkpoint_path=args.checkpoint_path,
        use_test_set=args.use_test_set,
        headless=headless
    )

    obs = evaluator.init()

    step = 0
    while step < max_step:
        if evaluator.policy:
            action = evaluator.policy.predict(observation=obs, deterministic=True)[0]
        else:
            action_dim = evaluator.env.action_space.shape[0]
            action = evaluator.env.action_space.sample()

        obs, reward, done, info = evaluator.step(action)

        cam_keys = ["faucet_viz", "faucet_viz2", "bucket_viz"]

        if save_img:
            for cam_key in cam_keys:
                if cam_key in obs:
                    rgb = obs[cam_key]  # shape: (H, W, 3)
                    img = Image.fromarray(rgb)
                    filename = f"{cam_key}_step_{step}.png"
                    img.save(os.path.join(img_save_path, filename))
                    # print(f"Saved: {filename}")
                else:
                    print(f"Warning: {cam_key} not found in obs!")

        if evaluator.env.is_eval_done:
            print(f"Evaluation success at step {step}.")
            break
        if done:
            print(f"Episode terminated at {step}, not successful.")
            break
        step += 1
        time.sleep(0.1)
    
    print("Demo eval completed.")