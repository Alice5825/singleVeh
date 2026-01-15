#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure Pursuit算法代价函数权重参数优化程序
功能：
1. 使用网格搜索和贝叶斯优化方法优化w1-w4参数
2. 运行仿真并收集性能指标
3. 分析结果并找出最优参数组合
4. 可视化参数与性能的关系2
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import os
import re
import time
import sys
from datetime import datetime
from sklearn.model_selection import ParameterGrid
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("警告: scikit-optimize未安装，将只使用网格搜索方法")

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

#迭代次数
CALLS = 10

# 解决Windows控制台编码问题
if sys.platform.startswith('win'):
    import io
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class WeightOptimizer:
    def __init__(self, script_path="pure_pursuit_multi_target.py"):
        """
        初始化权重优化器
        
        参数:
        script_path: 要运行的Python脚本路径
        """
        self.script_path = script_path
        self.results = []
        
        # 获取当前beta值
        beta_value = self._get_beta_value()
        
        # 创建以beta值和时刻命名的结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.result_dir = f"weight_optimization_results/beta_{beta_value}_{timestamp}"
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
            
        # 备份原始res.csv
        if os.path.exists("res.csv"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"res_backup_{timestamp}.csv"
            os.rename("res.csv", backup_name)
            print(f"已备份原始res.csv为 {backup_name}")
    
    def _get_beta_value(self):
        """
        从pure_pursuit_multi_target.py文件中获取BETA常量值
        
        返回:
        float: BETA常量值
        """
        try:
            # 读取pure_pursuit_multi_target.py文件内容
            with open(self.script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 使用正则表达式查找BETA值
            pattern = r'BETA\s*=\s*(\d+\.?\d*)'
            match = re.search(pattern, content)
            
            if match:
                beta_value = float(match.group(1))
                print(f"检测到BETA值: {beta_value}")
                return beta_value
            else:
                print("未找到BETA常量定义，使用默认值0.3")
                return 0.3
        except Exception as e:
            print(f"获取BETA值时发生异常: {e}，使用默认值0.3")
            return 0.3

    def modify_weights(self, w1, w2, w3, w4):
        """
        修改pure_pursuit_multi_target.py文件中PurePursuit类的四个代价权重参数（w1-w4）
        通过正则表达式定位并替换self.w1、self.w2、self.w3、self.w4的赋值语句
        修改完成后将新内容写回原文件，使后续仿真使用更新后的权重
        """
        """
        修改pure_pursuit_multi_target.py文件中的权重参数
        
        参数:
        w1, w2, w3, w4: 权重参数值
        """
        # 读取原文件内容
        with open(self.script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则表达式替换权重值
        replacements = [
            (r'self\.w1\s*=\s*\d+\.?\d*', f'self.w1 = {w1}'),
            (r'self\.w2\s*=\s*\d+\.?\d*', f'self.w2 = {w2}'),
            (r'self\.w3\s*=\s*\d+\.?\d*', f'self.w3 = {w3}'),
            (r'self\.w4\s*=\s*\d+\.?\d*', f'self.w4 = {w4}')
        ]
        
        modified_content = content
        for pattern, replacement in replacements:
            modified_content = re.sub(pattern, replacement, modified_content)
        
        # 写回文件
        with open(self.script_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"已将权重参数修改为 w1={w1}, w2={w2}, w3={w3}, w4={w4}")

    def run_simulation(self, weights):
        """
        运行仿真程序
        
        参数:
        weights: 权重参数元组 (w1, w2, w3, w4)
        
        返回:
        bool: 是否成功运行
        """
        w1, w2, w3, w4 = weights
        print(f"\n正在运行仿真，权重参数 = ({w1:.2f}, {w2:.2f}, {w3:.2f}, {w4:.2f})")
        
        # 备份当前res.csv文件（如果存在）
        res_csv_exists = os.path.exists("res.csv")
        if res_csv_exists:
            # 记录运行前的行数
            try:
                df_before = pd.read_csv("res.csv")
                rows_before = len(df_before)
            except:
                rows_before = 0
        else:
            rows_before = 0
        
        try:
            # 运行仿真程序，使用errors='ignore'来忽略编码错误
            result = subprocess.run(
                ["python", self.script_path],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',  # 忽略编码错误
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                print(f"仿真运行成功，权重参数 = ({w1:.2f}, {w2:.2f}, {w3:.2f}, {w4:.2f})")
                
                # 检查res.csv是否有新增数据
                if os.path.exists("res.csv"):
                    try:
                        df_after = pd.read_csv("res.csv")
                        rows_after = len(df_after)
                        
                        if rows_after > rows_before:
                            print(f"res.csv已更新，新增{rows_after - rows_before}行数据")
                        else:
                            print("警告：res.csv未更新或数据行数未增加")
                    except Exception as e:
                        print(f"检查res.csv更新时发生异常: {e}")
                
                return True
            else:
                print(f"仿真运行失败，权重参数 = ({w1:.2f}, {w2:.2f}, {w3:.2f}, {w4:.2f})")
                print("标准输出:")
                print(result.stdout[-1000:])  # 只显示最后1000个字符
                print("错误输出:")
                print(result.stderr[-1000:])  # 只显示最后1000个字符
                return False
                
        except subprocess.TimeoutExpired:
            print(f"仿真运行超时，权重参数 = ({w1:.2f}, {w2:.2f}, {w3:.2f}, {w4:.2f})")
            return False
        except Exception as e:
            print(f"运行仿真时发生异常，权重参数 = ({w1:.2f}, {w2:.2f}, {w3:.2f}, {w4:.2f}): {e}")
            print(f"异常类型: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False

    def extract_metrics(self, weights):
        """
        从res.csv中提取性能指标
        
        参数:
        weights: 权重参数元组 (w1, w2, w3, w4)
        
        返回:
        dict: 包含性能指标的字典
        """
        w1, w2, w3, w4 = weights
        try:
            # 读取res.csv文件
            if not os.path.exists("res.csv"):
                print("res.csv文件不存在")
                return None
                
            # 读取CSV文件
            df = pd.read_csv("res.csv")
            
            # 检查数据框是否为空
            if df.empty:
                print("res.csv文件为空")
                return None
            
            # 处理可能存在的重复标题行
            # 如果第一行是标题，确保使用正确的列名
            column_mapping = {
                'beta': 'beta',
                '平均横向误差': 'avg_lateral_error',
                '横向误差标准差': 'lateral_error_std',
                '转向角标准差': 'steering_angle_std'
            }
            
            # 检查是否需要重命名列（支持混合中英文列名）
            # 即使'beta'列存在，其他列也可能是中文，需要映射
            columns_to_rename = {}
            for col in df.columns:
                if col in column_mapping and column_mapping[col] != col:
                    columns_to_rename[col] = column_mapping[col]
            
            if columns_to_rename:
                df = df.rename(columns=columns_to_rename)
                print(f"已将列名映射为英文列名: {list(df.columns)}")
            
            # 去除重复的标题行（如果有的话）
            if 'beta' in df.columns:
                df = df[df['beta'] != 'beta']
            
            # 确保数值类型正确
            if 'beta' in df.columns:
                df['beta'] = pd.to_numeric(df['beta'], errors='coerce')
            if 'avg_lateral_error' in df.columns:
                df['avg_lateral_error'] = pd.to_numeric(df['avg_lateral_error'], errors='coerce')
            if 'lateral_error_std' in df.columns:
                df['lateral_error_std'] = pd.to_numeric(df['lateral_error_std'], errors='coerce')
            if 'steering_angle_std' in df.columns:
                df['steering_angle_std'] = pd.to_numeric(df['steering_angle_std'], errors='coerce')
            
            # 过滤掉无效行
            if 'beta' in df.columns:
                df = df.dropna(subset=['beta'])
            
            # 检查是否有数据
            if len(df) == 0:
                print("res.csv中无有效数据")
                return None
                
            # 获取最后一行数据作为当前权重下的性能指标
            try:
                last_row = df.iloc[-1]
            except IndexError:
                print("无法访问res.csv中的数据行")
                return None
            
            # 检查last_row是否包含所需的列
            required_columns = ['beta', 'avg_lateral_error', 'lateral_error_std', 'steering_angle_std']
            for col in required_columns:
                if col not in last_row:
                    print(f"res.csv中缺少必要的列: {col}")
                    print(f"当前可用列: {list(last_row.index)}")
                    return None
                if pd.isna(last_row[col]):
                    print(f"res.csv中列{col}的数据为NaN")
                    return None
                    
            # 检查数据是否有效
            try:
                beta_val = float(last_row['beta'])
                avg_lateral_error_val = float(last_row['avg_lateral_error'])
                lateral_error_std_val = float(last_row['lateral_error_std'])
                steering_angle_std_val = float(last_row['steering_angle_std'])
            except (ValueError, TypeError) as e:
                print(f"res.csv中的数据无法转换为浮点数: {e}")
                return None
            
            # 提取指标
            metrics = {
                'w1': w1,
                'w2': w2,
                'w3': w3,
                'w4': w4,
                'beta': beta_val,
                'avg_lateral_error': avg_lateral_error_val,
                'lateral_error_std': lateral_error_std_val,
                'steering_angle_std': steering_angle_std_val
            }
            
            print(f"提取到性能指标: {metrics}")
            return metrics
            
        except Exception as e:
            print(f"提取性能指标时发生异常: {e}")
            import traceback
            traceback.print_exc()
            return None

    def grid_search(self, param_ranges=None):
        """
        执行网格搜索优化
        
        参数:
        param_ranges: 参数范围字典，默认为:
            {
                'w1': [0.3, 0.7],
                'w2': [0.1, 0.5], 
                'w3': [0.1, 0.3],
                'w4': [0.1, 0.3]
            }
        """
        if param_ranges is None:
            param_ranges = {
                'w1': np.arange(0.3, 0.71, 0.1),  # [0.3, 0.4, 0.5, 0.6, 0.7]
                'w2': np.arange(0.1, 0.51, 0.1),  # [0.1, 0.2, 0.3, 0.4, 0.5]
                'w3': np.arange(0.1, 0.31, 0.1),  # [0.1, 0.2, 0.3]
                'w4': np.arange(0.1, 0.31, 0.1)   # [0.1, 0.2, 0.3]
            }
        
        print("开始网格搜索优化...")
        print(f"参数范围: {param_ranges}")
        
        # 生成参数组合
        param_grid = ParameterGrid(param_ranges)
        param_list = list(param_grid)
        
        print(f"总共需要测试 {len(param_list)} 种参数组合")
        
        # 遍历每个参数组合
        for i, params in enumerate(param_list):
            w1, w2, w3, w4 = params['w1'], params['w2'], params['w3'], params['w4']
            # 添加权重参数之和等于1的约束
            weight_sum = w1 + w2 + w3 + w4
            if abs(weight_sum - 1.0) > 1e-6:
                # 如果权重之和不为1，则进行归一化处理
                w1, w2, w3, w4 = w1/weight_sum, w2/weight_sum, w3/weight_sum, w4/weight_sum
                print(f"权重参数已归一化以满足和为1的约束: ({w1:.3f}, {w2:.3f}, {w3:.3f}, {w4:.3f})")
            
            print(f"\n[{i+1}/{len(param_list)}] 测试权重参数 = ({w1:.3f}, {w2:.3f}, {w3:.3f}, {w4:.3f})")
            if w1 < 0.3 or w1 > 0.7 or w2 < 0.1 or w2 > 0.5 or w3 < 0.1 or w3 > 0.3 or w4 < 0.1 or w4 > 0.3:
                continue
            # 修改权重参数
            self.modify_weights(w1, w2, w3, w4)
            
            # 运行仿真
            success = self.run_simulation((w1, w2, w3, w4))
            
            if success:
                # 提取性能指标
                metrics = self.extract_metrics((w1, w2, w3, w4))
                if metrics:
                    self.results.append(metrics)
                    
                    # 实时保存结果
                    self.save_results()
                    
                    # 绘制当前结果
                    self.plot_current_results()
            
            # 添加短暂延迟避免过快执行
            time.sleep(1)
        
        # 显示网格搜索的最优参数
        if self.results:
            # 转换为DataFrame
            df = pd.DataFrame(self.results)
            
            # 计算综合性能得分（归一化后相加）
            lateral_norm = (df['lateral_error_std'] - df['lateral_error_std'].min()) / (df['lateral_error_std'].max() - df['lateral_error_std'].min() + 1e-8)
            steering_norm = (df['steering_angle_std'] - df['steering_angle_std'].min()) / (df['steering_angle_std'].max() - df['steering_angle_std'].min() + 1e-8)
            df['综合得分'] = lateral_norm + steering_norm
            
            # 找出最优参数
            best_idx = df['综合得分'].idxmin()
            best_result = df.loc[best_idx]
            
            print("\n网格搜索完成!")
            print("=== 网格搜索最优参数 ===")
            print(f"最优权重参数:")
            print(f"  w1 (距离权重): {best_result['w1']:.3f}")
            print(f"  w2 (曲率权重): {best_result['w2']:.3f}")
            print(f"  w3 (方向权重): {best_result['w3']:.3f}")
            print(f"  w4 (横向误差权重): {best_result['w4']:.3f}")
            print(f"综合得分: {best_result['综合得分']:.6f}")
        else:
            print("\n网格搜索优化完成!")

    def bayesian_optimization(self, n_calls=50):
        """
        执行贝叶斯优化
        
        参数:
        n_calls: 优化迭代次数
        """
        if not SKOPT_AVAILABLE:
            print("错误: 未安装scikit-optimize库，无法执行贝叶斯优化")
            print("请安装: pip install scikit-optimize")
            return
            
        print("开始贝叶斯优化...")
        print(f"优化迭代次数: {n_calls}")
        
        # 定义搜索空间
        dimensions = [
            Real(0.3, 0.7, name='w1'),
            Real(0.1, 0.5, name='w2'),
            Real(0.1, 0.3, name='w3'),
            Real(0.1, 0.3, name='w4')
        ]
        
        # 定义目标函数
        @use_named_args(dimensions)
        def objective(w1, w2, w3, w4):
            # 添加权重参数之和等于1的约束
            weight_sum = w1 + w2 + w3 + w4
            if abs(weight_sum - 1.0) > 1e-6:
                # 如果权重之和不为1，则进行归一化处理
                w1, w2, w3, w4 = w1/weight_sum, w2/weight_sum, w3/weight_sum, w4/weight_sum
                print(f"权重参数已归一化以满足和为1的约束: ({w1:.3f}, {w2:.3f}, {w3:.3f}, {w4:.3f})")
            
            print(f"\n贝叶斯优化测试权重参数 = ({w1:.3f}, {w2:.3f}, {w3:.3f}, {w4:.3f})")
            
            # 修改权重参数
            self.modify_weights(w1, w2, w3, w4)
            
            # 运行仿真
            success = self.run_simulation((w1, w2, w3, w4))
            
            if success:
                # 提取性能指标
                metrics = self.extract_metrics((w1, w2, w3, w4))
                if metrics:
                    self.results.append(metrics)
                    
                    # 实时保存结果
                    self.save_results()
                    
                    # 计算目标函数值（越小越好）
                    # 使用综合性能得分作为目标函数
                    lateral_norm = metrics['lateral_error_std']
                    steering_norm = metrics['steering_angle_std']
                    objective_value = lateral_norm + steering_norm
                    
                    print(f"目标函数值: {objective_value:.6f}")
                    return objective_value
            
            # 如果运行失败，返回较大的值
            print("仿真运行失败，返回惩罚值")
            return 1000.0  # 惩罚值
        
        # 确保迭代次数满足最小要求
        if n_calls < 10:
            print(f"注意: 迭代次数{n_calls}小于推荐的最小值10，将使用10次迭代")
            n_calls = 10
            
        # 调整初始点数量以适应较小的迭代次数
        n_initial_points = min(10, n_calls // 2)
        if n_initial_points < 2:
            n_initial_points = 2
            
        # 执行贝叶斯优化
        try:
            # 开始贝叶斯优化，gp_minimize 会在内部循环调用 objective 函数 n_calls 次
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=n_calls,
                random_state=42,
                n_initial_points=n_initial_points
            )
            
            # 对最优参数进行归一化处理（与目标函数中的一致）
            w1_opt, w2_opt, w3_opt, w4_opt = result.x[0], result.x[1], result.x[2], result.x[3]
            weight_sum = w1_opt + w2_opt + w3_opt + w4_opt
            if abs(weight_sum - 1.0) > 1e-6:
                # 如果权重之和不为1，则进行归一化处理
                w1_opt, w2_opt, w3_opt, w4_opt = w1_opt/weight_sum, w2_opt/weight_sum, w3_opt/weight_sum, w4_opt/weight_sum
                print(f"注意: 最优参数已归一化以满足和为1的约束")
            
            print("\n贝叶斯优化完成!")
            print(f"最优参数: w1={w1_opt:.3f}, w2={w2_opt:.3f}, w3={w3_opt:.3f}, w4={w4_opt:.3f}")
            print(f"最优目标函数值: {result.fun:.6f}")
            
        except Exception as e:
            print(f"贝叶斯优化过程中发生异常: {e}")
            import traceback
            traceback.print_exc()

    def save_results(self):
        """
        保存优化结果到CSV文件
        """
        if not self.results:
            print("没有结果可保存")
            return
            
        # 转换为DataFrame
        df = pd.DataFrame(self.results)
        
        # 保存到CSV文件
        csv_path = os.path.join(self.result_dir, "weight_optimization_results.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"结果已保存到 {csv_path}")
        
        # 同时保存到主目录的res.csv
        df.to_csv("res.csv", index=False, encoding='utf-8-sig')
        print("结果已同步到 res.csv")

    def plot_current_results(self):
        """
        绘制当前优化结果
        """
        if len(self.results) < 2:
            return  # 至少需要2个点才能绘制图表
            
        # 转换为DataFrame
        df = pd.DataFrame(self.results)
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Pure Pursuit算法权重参数优化结果', fontsize=16)
        
        # w1 vs 平均横向误差
        scatter1 = axes[0, 0].scatter(df['w1'], df['avg_lateral_error'], c=df['lateral_error_std'], cmap='viridis')
        axes[0, 0].set_xlabel('w1 (距离权重)')
        axes[0, 0].set_ylabel('平均横向误差')
        axes[0, 0].grid(True)
        plt.colorbar(scatter1, ax=axes[0, 0], label='横向误差标准差')
        
        # w2 vs 平均横向误差
        scatter2 = axes[0, 1].scatter(df['w2'], df['avg_lateral_error'], c=df['lateral_error_std'], cmap='viridis')
        axes[0, 1].set_xlabel('w2 (曲率权重)')
        axes[0, 1].set_ylabel('平均横向误差')
        axes[0, 1].grid(True)
        plt.colorbar(scatter2, ax=axes[0, 1], label='横向误差标准差')
        
        # w3 vs 平均横向误差
        scatter3 = axes[0, 2].scatter(df['w3'], df['avg_lateral_error'], c=df['lateral_error_std'], cmap='viridis')
        axes[0, 2].set_xlabel('w3 (方向权重)')
        axes[0, 2].set_ylabel('平均横向误差')
        axes[0, 2].grid(True)
        plt.colorbar(scatter3, ax=axes[0, 2], label='横向误差标准差')
        
        # w4 vs 平均横向误差
        scatter4 = axes[1, 0].scatter(df['w4'], df['avg_lateral_error'], c=df['lateral_error_std'], cmap='viridis')
        axes[1, 0].set_xlabel('w4 (横向误差权重)')
        axes[1, 0].set_ylabel('平均横向误差')
        axes[1, 0].grid(True)
        plt.colorbar(scatter4, ax=axes[1, 0], label='横向误差标准差')
        
        # 横向误差标准差 vs 转向角标准差
        scatter5 = axes[1, 1].scatter(df['lateral_error_std'], df['steering_angle_std'], c=df['w1'], cmap='viridis')
        axes[1, 1].set_xlabel('横向误差标准差')
        axes[1, 1].set_ylabel('转向角标准差')
        axes[1, 1].grid(True)
        plt.colorbar(scatter5, ax=axes[1, 1], label='w1')
        
        # 综合性能指标（归一化后相加）
        # 归一化处理
        lateral_norm = (df['lateral_error_std'] - df['lateral_error_std'].min()) / (df['lateral_error_std'].max() - df['lateral_error_std'].min() + 1e-8)
        steering_norm = (df['steering_angle_std'] - df['steering_angle_std'].min()) / (df['steering_angle_std'].max() - df['steering_angle_std'].min() + 1e-8)
        composite_score = lateral_norm + steering_norm
        
        scatter6 = axes[1, 2].scatter(df['w1'], df['w2'], c=composite_score, cmap='viridis')
        axes[1, 2].set_xlabel('w1 (距离权重)')
        axes[1, 2].set_ylabel('w2 (曲率权重)')
        axes[1, 2].grid(True)
        plt.colorbar(scatter6, ax=axes[1, 2], label='综合性能得分 (越小越好)')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(self.result_dir, "optimization_progress.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"进度图已保存到 {plot_path}")
        
        # 显示当前最优参数
        best_idx = composite_score.idxmin()
        best_weights = df.loc[best_idx, ['w1', 'w2', 'w3', 'w4']]
        best_score = composite_score[best_idx]
        print(f"当前最优权重参数: w1={best_weights['w1']:.3f}, w2={best_weights['w2']:.3f}, w3={best_weights['w3']:.3f}, w4={best_weights['w4']:.3f}")
        print(f"综合得分: {best_score:.4f}")
        
        plt.close()

    def analyze_results(self):
        """
        分析优化结果并找出最优参数组合
        """
        if not self.results:
            print("没有结果可分析")
            return
            
        # 转换为DataFrame
        df = pd.DataFrame(self.results)
        
        # 计算综合性能得分（归一化后相加）
        lateral_norm = (df['lateral_error_std'] - df['lateral_error_std'].min()) / (df['lateral_error_std'].max() - df['lateral_error_std'].min() + 1e-8)
        steering_norm = (df['steering_angle_std'] - df['steering_angle_std'].min()) / (df['steering_angle_std'].max() - df['steering_angle_std'].min() + 1e-8)
        df['综合得分'] = lateral_norm + steering_norm
        
        # 找出最优参数
        best_idx = df['综合得分'].idxmin()
        best_result = df.loc[best_idx]
        
        print("\n=== 最优参数分析结果 ===")
        print(f"最优权重参数:")
        print(f"  w1 (距离权重): {best_result['w1']:.3f}")
        print(f"  w2 (曲率权重): {best_result['w2']:.3f}")
        print(f"  w3 (方向权重): {best_result['w3']:.3f}")
        print(f"  w4 (横向误差权重): {best_result['w4']:.3f}")
        print(f"平均横向误差: {best_result['avg_lateral_error']:.6f}")
        print(f"横向误差标准差: {best_result['lateral_error_std']:.6f}")
        print(f"转向角标准差: {best_result['steering_angle_std']:.6f}")
        print(f"综合得分: {best_result['综合得分']:.6f}")
        
        # 保存分析报告
        report_path = os.path.join(self.result_dir, "weight_optimization_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Pure Pursuit算法权重参数优化报告\n")
            f.write("=" * 40 + "\n")
            f.write(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"测试参数组合数: {len(df)}\n")
            f.write("\n最优参数:\n")
            f.write(f"w1 (距离权重): {best_result['w1']:.3f}\n")
            f.write(f"w2 (曲率权重): {best_result['w2']:.3f}\n")
            f.write(f"w3 (方向权重): {best_result['w3']:.3f}\n")
            f.write(f"w4 (横向误差权重): {best_result['w4']:.3f}\n")
            f.write(f"平均横向误差: {best_result['avg_lateral_error']:.6f}\n")
            f.write(f"横向误差标准差: {best_result['lateral_error_std']:.6f}\n")
            f.write(f"转向角标准差: {best_result['steering_angle_std']:.6f}\n")
            f.write(f"综合得分: {best_result['综合得分']:.6f}\n")
            
        print(f"分析报告已保存到 {report_path}")
        
        # 绘制最终结果图
        self.plot_final_results(df)

    def plot_final_results(self, df):
        """
        绘制最终优化结果图
        
        参数:
        df: 包含所有结果的DataFrame
        """
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Pure Pursuit算法权重参数优化最终结果', fontsize=18)
        
        # w1 vs 平均横向误差
        scatter1 = axes[0, 0].scatter(df['w1'], df['avg_lateral_error'], c=df['lateral_error_std'], cmap='viridis', s=60)
        axes[0, 0].set_xlabel('w1 (距离权重)', fontsize=12)
        axes[0, 0].set_ylabel('平均横向误差 (m)', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(labelsize=10)
        plt.colorbar(scatter1, ax=axes[0, 0], label='横向误差标准差')
        
        # w2 vs 平均横向误差
        scatter2 = axes[0, 1].scatter(df['w2'], df['avg_lateral_error'], c=df['lateral_error_std'], cmap='viridis', s=60)
        axes[0, 1].set_xlabel('w2 (曲率权重)', fontsize=12)
        axes[0, 1].set_ylabel('平均横向误差 (m)', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(labelsize=10)
        plt.colorbar(scatter2, ax=axes[0, 1], label='横向误差标准差')
        
        # w3 vs 平均横向误差
        scatter3 = axes[0, 2].scatter(df['w3'], df['avg_lateral_error'], c=df['lateral_error_std'], cmap='viridis', s=60)
        axes[0, 2].set_xlabel('w3 (方向权重)', fontsize=12)
        axes[0, 2].set_ylabel('平均横向误差 (m)', fontsize=12)
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].tick_params(labelsize=10)
        plt.colorbar(scatter3, ax=axes[0, 2], label='横向误差标准差')
        
        # w4 vs 平均横向误差
        scatter4 = axes[1, 0].scatter(df['w4'], df['avg_lateral_error'], c=df['lateral_error_std'], cmap='viridis', s=60)
        axes[1, 0].set_xlabel('w4 (横向误差权重)', fontsize=12)
        axes[1, 0].set_ylabel('平均横向误差 (m)', fontsize=12)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(labelsize=10)
        plt.colorbar(scatter4, ax=axes[1, 0], label='横向误差标准差')
        
        # 横向误差标准差 vs 转向角标准差
        scatter5 = axes[1, 1].scatter(df['lateral_error_std'], df['steering_angle_std'], c=df['综合得分'], cmap='viridis', s=60)
        axes[1, 1].set_xlabel('横向误差标准差 (m)', fontsize=12)
        axes[1, 1].set_ylabel('转向角标准差 (度)', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(labelsize=10)
        plt.colorbar(scatter5, ax=axes[1, 1], label='综合得分')
        
        # 综合性能得分
        scatter6 = axes[1, 2].scatter(df['w1'], df['w2'], c=df['综合得分'], cmap='viridis', s=60)
        axes[1, 2].set_xlabel('w1 (距离权重)', fontsize=12)
        axes[1, 2].set_ylabel('w2 (曲率权重)', fontsize=12)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].tick_params(labelsize=10)
        plt.colorbar(scatter6, ax=axes[1, 2], label='综合性能得分 (越小越好)')
        
        # 标记最优参数点
        best_idx = df['综合得分'].idxmin()
        best_w1 = df.loc[best_idx, 'w1']
        best_w2 = df.loc[best_idx, 'w2']
        
        for ax in axes.flat:
            ax.scatter(best_w1, best_w2, color='red', marker='*', s=200, zorder=5)
            ax.text(best_w1, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1, 
                   f'最优: ({best_w1:.3f}, {best_w2:.3f})', 
                   fontsize=10, color='red', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        plot_path = os.path.join(self.result_dir, "final_weight_optimization_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"最终结果图已保存到 {plot_path}")
        
        plt.close()

def main():
    """主函数"""
    print("Pure Pursuit算法权重参数优化程序")
    print("=" * 40)
    
    # 获取用户选择的优化方法
    print("请选择优化方法:")
    print("1. 网格搜索")
    print("2. 贝叶斯优化")

    choice = 1
    # 创建权重优化器实例
    optimizer = WeightOptimizer()
    
    # 执行优化
    if choice == 1:
        # 网格搜索
        optimizer.grid_search()
    elif choice == 2:
        # 贝叶斯优化
        if SKOPT_AVAILABLE:
            try:
                n_calls = CALLS
                optimizer.bayesian_optimization(n_calls=n_calls)
            except ValueError:
                print("输入格式错误，使用默认迭代次数50")
                optimizer.bayesian_optimization(n_calls=50)
        else:
            print("未安装scikit-optimize库，请先安装: pip install scikit-optimize")
            return
    
    # 分析结果
    optimizer.analyze_results()
    
    print("\n程序执行完成!")
    print(f"优化结果已保存至 {os.path.abspath(optimizer.result_dir)} 目录")
    print(f"该目录以beta值和时间戳命名，格式为: weight_optimization_results_beta_[beta值]_[时间戳]")

if __name__ == "__main__":
    main()