# MultiAgents

## 项目描述

这是一个基于多代理系统的6G+AI项目，旨在通过智能代理优化网络策略和资源分配。项目集成了意图编码、策略优化等功能，支持动态网络环境下的决策制定。

## 主要功能

- **意图编码代理 (IntentEncodingAgent)**: 处理用户意图的编码和解析
- **优化策略代理 (OptimizationStrategyAgent)**: 执行网络优化策略
- **策略模型**: 包括SM策略决策、UE上下文管理等
- **可视化工具**: 通过Streamlit应用提供用户界面
- **网络状态监控**: 实时监控网络状态和性能

## 项目结构

```
.
├── main.py                 # 主入口文件
├── streamlit_app.py        # Streamlit Web应用
├── pyproject.toml          # 项目配置和依赖
├── model/                  # 数据模型
│   ├── Arp.py
│   ├── PolicyTrigger.py
│   ├── RatType.py
│   ├── SmPolicyContextData.py
│   ├── SmPolicyDecision.py
│   ├── UeContext.py
│   ├── UrspRuleRequest.py
│   └── UserLocation.py
├── multi_agents/           # 多代理系统
│   ├── basemodel.py
│   ├── IntentEncodingAgent.py
│   └── OptimizationStrategyAgent.py
├── tools/                  # 工具模块
│   ├── network_status.py
│   ├── optimization_visable.py
│   └── optimization.py
├── optimizer/              # 优化器
│   ├── data.py
│   ├── engine.py
│   ├── interface.py
│   └── models.py
├── tests/                  # 测试文件
│   ├── test_iea.py
│   ├── test_optimization.py
│   └── test_osa.py
└── utils/                  # 工具函数
    └── logger.py
```

## 安装

1. 克隆项目：
   ```bash
   git clone <repository-url>
   cd MuiltiAgents
   ```

2. 创建虚拟环境：
   ```bash
   python -m venv .venv
   ```

3. 激活虚拟环境：
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```

4. 安装依赖：
   ```bash
   pip install -e .
   ```

## 使用

### 运行主程序
```bash
python main.py
```

### 运行Streamlit应用
```bash
streamlit run streamlit_app.py
```

### 运行测试
```bash
python -m pytest tests/
```

## 配置

项目使用`pyproject.toml`管理依赖和配置。主要依赖包括：
- streamlit: Web界面
- pytest: 测试框架
- 其他相关库（根据实际依赖调整）uv
