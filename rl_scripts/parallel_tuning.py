import os
import subprocess
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

class MultiGPUTuner:
    def __init__(self, gpus=[0,1,2,3,4,5,6,7]):
        self.gpus = gpus
        self.results = []
        self.best_config = None
        self.best_reward = float('-inf')
    
    def create_param_grid(self):
        """创建参数网格 - 分阶段调参"""
        
        # 阶段1: 核心参数粗调
        stage1_grid = {
            'learning_rate': [5e-4, 1e-3, 2e-3, 3e-3],
            'n_steps': [4096, 8192, 16384],
            'batch_size': [1024, 2048, 4096],
            'n_epochs': [10, 20, 30, 40],
            'ent_coef': [0.0, 0.01, 0.1, 0.5],
            'gamma': [0.95, 0.98, 0.99, 0.952],
            'clip_range': [0.1, 0.148, 0.2, 0.3]
        }
        
        # 生成所有参数组合
        param_names = list(stage1_grid.keys())
        param_values = list(stage1_grid.values())
        
        configs = []
        for i, combination in enumerate(itertools.product(*param_values)):
            config = dict(zip(param_names, combination))
            # 确保batch_size不超过n_steps
            if config['batch_size'] <= config['n_steps']:
                config['exp_id'] = i
                configs.append(config)
        
        return configs
    
    def run_single_experiment(self, config, gpu_id):
        """在指定GPU上运行单个实验"""
        exp_id = config['exp_id']
        
        # 创建实验目录
        exp_dir = f"/home/scratch/jiayuc2/tuning_results/exp_{exp_id}_gpu_{gpu_id}"
        os.makedirs(exp_dir, exist_ok=True)
        
        # 构建命令
        cmd = [
            "python", "run_ppo.py",
            "--learning-rate", str(config['learning_rate']),
            "--n-steps", str(config['n_steps']),
            "--batch-size", str(config['batch_size']),
            "--n-epochs", str(config['n_epochs']),
            "--ent-coef", str(config['ent_coef']),
            "--gamma", str(config['gamma']),
            "--clip-range", str(config['clip_range']),
            "--total-timesteps", "500000",  # 较快的测试
            "--cuda_id", str(gpu_id),
            "--seed", str(42 + exp_id),  # 不同的随机种子
            "--output-dir", exp_dir,
            "--eval-freq", "50000",
            "--save-freq", "250000"
        ]
        
        print(f"GPU {gpu_id} 开始实验 {exp_id}: LR={config['learning_rate']}, "
              f"Steps={config['n_steps']}, Batch={config['batch_size']}")
        
        try:
            # 运行实验
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=3600,  # 1小时超时
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            
            # 解析结果
            reward = self.parse_result(exp_dir, result.stdout)
            
            experiment_result = {
                'exp_id': exp_id,
                'gpu_id': gpu_id,
                'config': config,
                'final_reward': reward,
                'success': reward > -999,
                'output_dir': exp_dir
            }
            
            print(f"GPU {gpu_id} 完成实验 {exp_id}: 奖励 = {reward:.3f}")
            return experiment_result
            
        except subprocess.TimeoutExpired:
            print(f"GPU {gpu_id} 实验 {exp_id} 超时")
            return {
                'exp_id': exp_id,
                'gpu_id': gpu_id,
                'config': config,
                'final_reward': -999,
                'success': False,
                'error': 'timeout'
            }
        except Exception as e:
            print(f"GPU {gpu_id} 实验 {exp_id} 出错: {e}")
            return {
                'exp_id': exp_id,
                'gpu_id': gpu_id,
                'config': config,
                'final_reward': -999,
                'success': False,
                'error': str(e)
            }
    
    def parse_result(self, exp_dir, stdout):
        """解析实验结果"""
        try:
            # 方法1: 从evaluations文件读取
            eval_file = os.path.join(exp_dir, "evaluations", "evaluations.npz")
            if os.path.exists(eval_file):
                import numpy as np
                data = np.load(eval_file)
                if 'results' in data and len(data['results']) > 0:
                    return float(data['results'][-1])  # 最后一次评估结果
            
            # 方法2: 从TensorBoard日志读取
            tb_dir = os.path.join(exp_dir, "tb")
            if os.path.exists(tb_dir):
                # 这里可以添加TensorBoard日志解析
                pass
            
            # 方法3: 从stdout解析（需要修改run_ppo.py添加输出）
            if "Final evaluation reward:" in stdout:
                lines = stdout.split('\n')
                for line in lines:
                    if "Final evaluation reward:" in line:
                        return float(line.split(':')[-1].strip())
            
            return -100  # 默认惩罚值
            
        except Exception as e:
            print(f"解析结果出错: {e}")
            return -999
    
    def run_parallel_tuning(self, max_experiments=None):
        """并行运行调参"""
        configs = self.create_param_grid()
        
        if max_experiments:
            configs = configs[:max_experiments]
        
        print(f"总共{len(configs)}个实验配置")
        print(f"使用{len(self.gpus)}块GPU并行运行")
        
        # 使用进程池并行运行
        with ProcessPoolExecutor(max_workers=len(self.gpus)) as executor:
            # 提交所有任务
            future_to_config = {}
            gpu_cycle = itertools.cycle(self.gpus)
            
            for config in configs:
                gpu_id = next(gpu_cycle)
                future = executor.submit(self.run_single_experiment, config, gpu_id)
                future_to_config[future] = config
            
            # 收集结果
            for future in as_completed(future_to_config):
                result = future.result()
                self.results.append(result)
                
                # 更新最佳配置
                if result['final_reward'] > self.best_reward:
                    self.best_reward = result['final_reward']
                    self.best_config = result['config']
                    print(f"🏆 新的最佳配置! 奖励: {self.best_reward:.3f}")
                    print(f"   参数: {self.best_config}")
        
        return self.results
    
    def save_results(self, filename="tuning_results.json"):
        """保存调参结果"""
        results_summary = {
            'best_config': self.best_config,
            'best_reward': self.best_reward,
            'all_results': self.results,
            'total_experiments': len(self.results),
            'successful_experiments': len([r for r in self.results if r['success']])
        }
        
        with open(filename, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"结果已保存到 {filename}")
        return results_summary

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-experiments", type=int, default=None, 
                       help="限制实验数量（用于测试）")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0,1,2,3,4,5,6,7],
                       help="使用的GPU ID列表")
    args = parser.parse_args()
    
    tuner = MultiGPUTuner(gpus=args.gpus)
    results = tuner.run_parallel_tuning(max_experiments=args.max_experiments)
    summary = tuner.save_results()
    
    print(f"\n{'='*50}")
    print("调参完成!")
    print(f"最佳奖励: {summary['best_reward']:.3f}")
    print(f"最佳配置: {summary['best_config']}")
    print(f"成功实验: {summary['successful_experiments']}/{summary['total_experiments']}")