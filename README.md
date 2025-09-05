# 2025-mcm

## 架构

项目采用C++与Python混合编程架构：

- **C++模拟函数**：使用C++编写物理仿真函数，通过pybind11提供Python接口
- **Python优化算法**：实现贝叶斯优化和粒子群优化算法进行参数寻优

### 主要文件说明

- `cpp/libsimulate.cc`：C++仿真核心代码
- `compile.py`：C++模块编译脚本
- `object.py`：定义导弹、无人机、烟幕等对象类
- `main.py`：问题1的求解程序
- `bayes.py`：使用贝叶斯优化求解问题2和3
- `pso.py`：使用粒子群优化求解问题4和5

## 环境要求

### 系统要求

- Python >= 3.13
- C++编译器（支持C++17）

### 依赖管理

本项目使用[uv](https://github.com/astral-sh/uv)作为Python包管理器，依赖项均在`pyproject.toml`文件中

## 安装与运行

### 1. 安装uv包管理器

```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 安装项目依赖

```bash
uv sync
```

### 3. 编译C++模拟模块

```bash
uv run compile.py
```

### 4. 运行求解程序

#### 问题1：基础遮蔽时长分析

```bash
uv run main.py
```

#### 问题2和3：贝叶斯优化求解

```bash
uv run bayes.py
```

#### 问题4和5：粒子群优化求解

```bash
uv run pso.py
```