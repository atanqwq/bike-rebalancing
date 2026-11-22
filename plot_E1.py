import pandas as pd
import matplotlib.pyplot as plt
import os

# 设置中文字体，防止图表中的中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("开始绘制 Fig.2 (E1 聚类选择图)...")
    
    # 1. 读取数据
    k_df = pd.read_csv('results/E1_k_selection.csv')
    
    # 2. 创建画布 (1行2列的子图)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 找到最优的 k (轮廓系数最大的)
    best_k = k_df.loc[k_df['silhouette'].idxmax(), 'k']
    
    # 图 1: 肘部法则 (SSE)
    axes[0].plot(k_df['k'], k_df['sse'], marker='o', linestyle='-', color='b')
    axes[0].set_title('肘部法则 (Elbow Method)')
    axes[0].set_xlabel('簇的数量 (k)')
    axes[0].set_ylabel('误差平方和 (SSE)')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    # 标出最优 k 的位置
    axes[0].axvline(x=best_k, color='r', linestyle='--', label=f'最优 k={best_k}')
    axes[0].legend()
    
    # 图 2: 轮廓系数 (Silhouette)
    axes[1].plot(k_df['k'], k_df['silhouette'], marker='s', linestyle='-', color='g')
    axes[1].set_title('轮廓系数 (Silhouette Score)')
    axes[1].set_xlabel('簇的数量 (k)')
    axes[1].set_ylabel('轮廓系数')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    # 标出最优 k 的位置
    axes[1].axvline(x=best_k, color='r', linestyle='--', label=f'最优 k={best_k}')
    axes[1].legend()
    
    # 调整布局
    plt.tight_layout()
    
    # 3. 保存图片到 figures 文件夹
    os.makedirs('figures', exist_ok=True)
    save_path_png = 'figures/fig2_clusters.png'
    save_path_pdf = 'figures/fig2_clusters.pdf'
    
    # 按照文档要求，保存为 300 dpi
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight')
    
    print(f"✅ 绘图成功！图片已保存至: {save_path_png} 和 {save_path_pdf}")
    plt.show()

if __name__ == "__main__":
    main()