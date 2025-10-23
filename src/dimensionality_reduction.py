import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def apply_pca(X, n_components=None, scaler=None, random_state=42):
    """
    Áp dụng thuật toán PCA để giảm chiều dữ liệu.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào cần giảm chiều.
    n_components : int or None, default=None
        Số lượng thành phần chính cần giữ lại. Nếu None, giữ lại tất cả.
    scaler : sklearn.preprocessing.StandardScaler or None, default=None
        Đối tượng StandardScaler đã được fit. Nếu None, một StandardScaler mới sẽ được tạo.
    random_state : int, default=42
        Seed cho quá trình sinh số ngẫu nhiên để đảm bảo kết quả reproducible.
        
    Returns:
    --------
    X_pca : array-like, shape (n_samples, n_components)
        Dữ liệu sau khi áp dụng PCA.
    pca : sklearn.decomposition.PCA
        Đối tượng PCA đã được fit.
    scaler : sklearn.preprocessing.StandardScaler
        Đối tượng StandardScaler đã được fit.
    """
    # Chuẩn hóa dữ liệu
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Áp dụng PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, pca, scaler


def plot_explained_variance(pca, ax=None, title="Explained Variance Ratio"):
    """
    Vẽ biểu đồ tỉ lệ phương sai giải thích được.
    
    Parameters:
    -----------
    pca : sklearn.decomposition.PCA
        Đối tượng PCA đã được fit.
    ax : matplotlib.axes.Axes or None, default=None
        Đối tượng axes để vẽ. Nếu None, một axes mới sẽ được tạo.
    title : str, default="Explained Variance Ratio"
        Tiêu đề của biểu đồ.
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        Đối tượng axes đã được vẽ.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Tỉ lệ phương sai giải thích được
    explained_var_ratio = pca.explained_variance_ratio_
    cum_explained_var = np.cumsum(explained_var_ratio)
    
    # Vẽ biểu đồ
    components = range(1, len(explained_var_ratio) + 1)
    ax.bar(components, explained_var_ratio, alpha=0.8, label='Individual')
    ax.step(components, cum_explained_var, where='mid', label='Cumulative')
    ax.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90% Threshold')
    ax.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95% Threshold')
    
    # Xác định số lượng thành phần để giải thích 90% và 95% phương sai
    n_components_90 = np.argmax(cum_explained_var >= 0.9) + 1
    n_components_95 = np.argmax(cum_explained_var >= 0.95) + 1
    
    ax.axvline(x=n_components_90, color='r', linestyle=':', alpha=0.5)
    ax.axvline(x=n_components_95, color='g', linestyle=':', alpha=0.5)
    
    # Cấu hình biểu đồ
    ax.set_xlabel('Số thành phần chính')
    ax.set_ylabel('Tỉ lệ phương sai giải thích được')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Thông tin về số lượng thành phần
    textstr = f'90% variance: {n_components_90} components\n95% variance: {n_components_95} components'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)
    
    return ax


def apply_tsne(X, n_components=2, perplexity=30, random_state=42, scaler=None):
    """
    Áp dụng thuật toán t-SNE để giảm chiều dữ liệu.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào cần giảm chiều.
    n_components : int, default=2
        Số chiều đầu ra, thường là 2 hoặc 3 để có thể trực quan hóa.
    perplexity : float, default=30
        Liên quan đến số lượng láng giềng gần nhất. Thường chọn trong khoảng từ 5 đến 50.
    random_state : int, default=42
        Seed cho quá trình sinh số ngẫu nhiên để đảm bảo kết quả reproducible.
    scaler : sklearn.preprocessing.StandardScaler or None, default=None
        Đối tượng StandardScaler đã được fit. Nếu None, một StandardScaler mới sẽ được tạo.
        
    Returns:
    --------
    X_tsne : array-like, shape (n_samples, n_components)
        Dữ liệu sau khi áp dụng t-SNE.
    scaler : sklearn.preprocessing.StandardScaler
        Đối tượng StandardScaler đã được fit.
    """
    # Chuẩn hóa dữ liệu
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Áp dụng t-SNE
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                random_state=random_state, n_iter=1000, learning_rate=200)
    X_tsne = tsne.fit_transform(X_scaled)
    
    return X_tsne, scaler


def plot_2d_data(X, y=None, ax=None, title="Dữ liệu 2 chiều", labels=None, palette="viridis",
                 scatter_kwargs=None):
    """
    Vẽ dữ liệu 2 chiều với màu sắc theo nhãn nếu có.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 2)
        Dữ liệu 2 chiều cần vẽ.
    y : array-like, shape (n_samples,) or None, default=None
        Nhãn của từng điểm dữ liệu. Nếu None, tất cả các điểm sẽ có cùng màu.
    ax : matplotlib.axes.Axes or None, default=None
        Đối tượng axes để vẽ. Nếu None, một axes mới sẽ được tạo.
    title : str, default="Dữ liệu 2 chiều"
        Tiêu đề của biểu đồ.
    labels : list or None, default=None
        Danh sách các nhãn tương ứng với các giá trị trong y.
        Nếu None, các nhãn sẽ được hiển thị là các giá trị trong y.
    palette : str or list, default="viridis"
        Bảng màu hoặc danh sách màu để sử dụng.
    scatter_kwargs : dict or None, default=None
        Các tham số bổ sung cho hàm scatter.
        
    Returns:
    --------
    ax : matplotlib.axes.Axes
        Đối tượng axes đã được vẽ.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if scatter_kwargs is None:
        scatter_kwargs = {}
    
    default_kwargs = {
        'alpha': 0.7,
        's': 50,
        'edgecolor': 'k',
        'linewidth': 0.5
    }
    
    # Kết hợp tham số mặc định với tham số được cung cấp
    for key, value in default_kwargs.items():
        if key not in scatter_kwargs:
            scatter_kwargs[key] = value
    
    # Vẽ dữ liệu
    if y is None:
        ax.scatter(X[:, 0], X[:, 1], **scatter_kwargs)
    else:
        # Chuyển đổi y thành mảng nếu là series
        if hasattr(y, 'values'):
            y = y.values
            
        # Nếu y là categorical, chuyển đổi thành số
        if hasattr(y, 'dtype') and y.dtype.name == 'category':
            y = y.codes
            
        # Vẽ scatter plot với màu sắc theo nhãn
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=palette, **scatter_kwargs)
        
        # Thêm legend nếu có nhãn
        if labels is not None:
            # Tạo legend với các màu và nhãn tương ứng
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                  markersize=10) for i in range(len(labels))]
            ax.legend(handles, labels, loc='best', title='Phân loại')
            
    # Cấu hình biểu đồ
    ax.set_title(title)
    ax.set_xlabel('Thành phần 1')
    ax.set_ylabel('Thành phần 2')
    ax.grid(True, alpha=0.3)
    
    return ax


def find_optimal_components(X, variance_threshold=0.95, max_components=None, random_state=42):
    """
    Tìm số lượng thành phần tối ưu để giữ lại một lượng phương sai nhất định.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào.
    variance_threshold : float, default=0.95
        Ngưỡng phương sai cần giữ lại (từ 0 đến 1).
    max_components : int or None, default=None
        Số lượng thành phần tối đa cần xem xét. Nếu None, tất cả các thành phần sẽ được xem xét.
    random_state : int, default=42
        Seed cho quá trình sinh số ngẫu nhiên.
        
    Returns:
    --------
    n_components : int
        Số lượng thành phần tối ưu.
    explained_var : float
        Tỉ lệ phương sai giải thích được với số lượng thành phần tối ưu.
    """
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Xác định số lượng thành phần tối đa
    n_features = X_scaled.shape[1]
    if max_components is None or max_components > n_features:
        max_components = n_features
    
    # Áp dụng PCA
    pca = PCA(random_state=random_state)
    pca.fit(X_scaled)
    
    # Tìm số lượng thành phần để đạt ngưỡng phương sai
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    # Đảm bảo không vượt quá số lượng thành phần tối đa
    n_components = min(n_components, max_components)
    
    # Phương sai giải thích được với số lượng thành phần tối ưu
    explained_var = cumulative_variance[n_components - 1]
    
    return n_components, explained_var


def visualize_pca_components(X, feature_names=None, n_components=2, scaler=None, figsize=(12, 10)):
    """
    Trực quan hóa đóng góp của các đặc trưng vào các thành phần chính.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào.
    feature_names : list or None, default=None
        Danh sách tên các đặc trưng. Nếu None, các đặc trưng sẽ được đặt tên theo chỉ số.
    n_components : int, default=2
        Số lượng thành phần chính cần trực quan hóa.
    scaler : sklearn.preprocessing.StandardScaler or None, default=None
        Đối tượng StandardScaler đã được fit. Nếu None, một StandardScaler mới sẽ được tạo.
    figsize : tuple, default=(12, 10)
        Kích thước của biểu đồ.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Đối tượng figure đã được vẽ.
    """
    # Chuẩn hóa dữ liệu
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Áp dụng PCA
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    
    # Tạo tên đặc trưng nếu không được cung cấp
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # Tạo DataFrame cho các loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )
    
    # Tạo hình vẽ
    fig, axes = plt.subplots(n_components, 1, figsize=figsize)
    if n_components == 1:
        axes = [axes]
    
    # Vẽ biểu đồ thanh cho từng thành phần
    for i, ax in enumerate(axes):
        pc = loadings.columns[i]
        sorted_loadings = loadings[pc].sort_values()
        sorted_loadings.plot(kind='barh', ax=ax)
        ax.set_title(f'Loadings của {pc}')
        ax.set_xlabel('Độ lớn của loading')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return fig


def pca_biplot(X, y=None, feature_names=None, scaler=None, n_components=2, 
               labels=None, palette="viridis", figsize=(12, 10), random_state=42):
    """
    Tạo biplot để trực quan hóa dữ liệu và các đặc trưng trong không gian PCA.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Dữ liệu đầu vào.
    y : array-like, shape (n_samples,) or None, default=None
        Nhãn của từng điểm dữ liệu. Nếu None, tất cả các điểm sẽ có cùng màu.
    feature_names : list or None, default=None
        Danh sách tên các đặc trưng. Nếu None, các đặc trưng sẽ được đặt tên theo chỉ số.
    scaler : sklearn.preprocessing.StandardScaler or None, default=None
        Đối tượng StandardScaler đã được fit. Nếu None, một StandardScaler mới sẽ được tạo.
    n_components : int, default=2
        Số lượng thành phần chính cần trực quan hóa.
    labels : list or None, default=None
        Danh sách các nhãn tương ứng với các giá trị trong y.
    palette : str or list, default="viridis"
        Bảng màu hoặc danh sách màu để sử dụng.
    figsize : tuple, default=(12, 10)
        Kích thước của biểu đồ.
    random_state : int, default=42
        Seed cho quá trình sinh số ngẫu nhiên.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Đối tượng figure đã được vẽ.
    """
    # Chuẩn hóa dữ liệu
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Áp dụng PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    # Tạo tên đặc trưng nếu không được cung cấp
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X.shape[1])]
    
    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=figsize)
    
    # Vẽ điểm dữ liệu
    scatter_kwargs = {
        'alpha': 0.7,
        's': 60,
        'edgecolor': 'k',
        'linewidth': 0.5
    }
    
    if y is None:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], **scatter_kwargs)
    else:
        # Chuyển đổi y thành mảng nếu là series
        if hasattr(y, 'values'):
            y = y.values
        
        # Nếu y là categorical, chuyển đổi thành số
        if hasattr(y, 'dtype') and y.dtype.name == 'category':
            y = y.codes
            
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=palette, **scatter_kwargs)
        
        # Thêm legend nếu có nhãn
        if labels is not None:
            # Tạo legend với các màu và nhãn tương ứng
            handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=scatter.cmap(scatter.norm(i)), 
                                markersize=10) for i in range(len(labels))]
            ax.legend(handles, labels, loc='best', title='Phân loại')
    
    # Vẽ các vector đặc trưng
    feature_vectors = pca.components_.T[:, :2]
    arrow_scale = 5  # Tỉ lệ để vẽ các vector
    
    # Tính toán giới hạn của trục
    x_min, x_max = X_pca[:, 0].min(), X_pca[:, 0].max()
    y_min, y_max = X_pca[:, 1].min(), X_pca[:, 1].max()
    
    for i, (name, vec) in enumerate(zip(feature_names, feature_vectors)):
        ax.arrow(0, 0, arrow_scale * vec[0], arrow_scale * vec[1], 
                 head_width=0.2, head_length=0.2, fc='red', ec='red', alpha=0.7)
        text_pos = arrow_scale * 1.1 * vec
        ax.text(text_pos[0], text_pos[1], name, color='red', fontsize=10, ha='center', va='center')
    
    # Cấu hình biểu đồ
    ax.set_title('Biplot PCA')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} phương sai)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} phương sai)')
    
    # Đảm bảo tỉ lệ trục x và y giống nhau
    ax.set_aspect('equal')
    
    # Điều chỉnh giới hạn của trục để đảm bảo nhìn thấy tất cả các vector
    max_extent = max(
        abs(x_min), abs(x_max),
        abs(y_min), abs(y_max),
        abs(arrow_scale)
    )
    ax.set_xlim(-max_extent * 1.2, max_extent * 1.2)
    ax.set_ylim(-max_extent * 1.2, max_extent * 1.2)
    
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # Thêm thông tin về phương sai giải thích được
    textstr = f'PC1: {pca.explained_variance_ratio_[0]:.2%}\nPC2: {pca.explained_variance_ratio_[1]:.2%}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    return fig