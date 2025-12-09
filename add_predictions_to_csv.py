"""
将Model4 (最佳模型) 的FOS预测值添加到1201.csv

Model4使用 渗流自然电位 + 耦合电场 预测 渗流FOS
最佳算法: SVR (R² = 0.911)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from pathlib import Path

# 路径配置
DATA_DIR = Path(__file__).parent / 'data'
INPUT_FILE = DATA_DIR / '1201.csv'
OUTPUT_FILE = DATA_DIR / '1201.csv'  # 覆盖原文件

def main():
    print("=" * 70)
    print("Adding Model4 FOS Predictions to 1201.csv")
    print("=" * 70)
    
    # 加载数据
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    print(f"✓ Loaded {len(df)} records from {INPUT_FILE.name}")
    
    # Model4 配置
    feature_cols = [
        'Seepage Self-Potential Minimum',
        'Seepage Self-Potential Maximum',
        'Coupling Electric Field Minimum',
        'Coupling Electric Field Maximum'
    ]
    target_col = 'Seepage FOS Maximum'
    
    # 验证列存在
    missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing_cols:
        print(f"✗ Missing columns: {missing_cols}")
        return
    
    # 准备有效数据
    df_valid = df[feature_cols + [target_col]].dropna()
    valid_indices = df_valid.index
    print(f"✓ Valid samples: {len(df_valid)}")
    
    X = df_valid[feature_cols].values
    y = df_valid[target_col].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 使用SVR训练模型 (Model4最佳算法)
    print("\n训练SVR模型...")
    model = SVR(kernel='rbf', C=1.0)
    model.fit(X_scaled, y)
    
    # 对全部有效数据进行预测
    y_pred = model.predict(X_scaled)
    
    # 计算预测误差
    residuals = y - y_pred
    
    # 添加新列
    df['Predicted_FOV'] = np.nan
    df['FOV_Prediction_Residual'] = np.nan
    
    df.loc[valid_indices, 'Predicted_FOV'] = y_pred
    df.loc[valid_indices, 'FOV_Prediction_Residual'] = residuals
    
    # 计算并打印统计信息
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"\n模型性能 (全数据集):")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  Mean Residual = {residuals.mean():.4f}")
    print(f"  Std Residual = {residuals.std():.4f}")
    
    # 保存
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✓ Saved to {OUTPUT_FILE}")
    print(f"  New columns added:")
    print(f"    - Predicted_Seepage_FOS: Model4 SVR预测的渗流FOS")
    print(f"    - FOS_Prediction_Residual: 实际值 - 预测值")
    
    # 显示前几行验证
    print(f"\n验证 (前5行):")
    print(df[['Seepage FOS Maximum', 'Predicted_Seepage_FOS', 'FOS_Prediction_Residual']].head())

if __name__ == "__main__":
    main()
