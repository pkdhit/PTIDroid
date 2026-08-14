# Assertion Generation Framework README —— PTIDroid

## 1. 项目定位

本项目支持两种模型接入方式：

1. **云端 OpenAI 兼容 API**：通过 `base_url + api_key` 调用外部大模型。
2. **本地服务器模型**：通过 `base_url` 访问服务器上启动的 Qwen3-VL OpenAI-compatible 服务。

---

## 2. 框架设计思路

整个断言生成流程分为五个核心板块：

- **Assertion Classifier**
- **Intent Planner**
- **Control Selector**
- **Assertion Generator**
- **Assertion Checker**

### 2.1 模块职责

#### Assertion Classifier
- 输入：无断言测试用例的自然语言描述（`TaskDescription`）。
- 输出：断言类型类别编号（1~4）。
- 作用：为后续阶段规划和组件筛选提供语义类别，不直接生成断言。
- 消融基线：可以通过 `--ablation no_classifier` 或批处理脚本的 `--no-classifier` 关闭分类器，此时后续流程会使用 `category=None` 和通用断言集合，不再依赖分类结果。

四类断言分别是：

1. **Simple Functional Check Assertion**
2. **UI Hierarchy-based Assertions**
3. **Interaction / Logic Assertions**
4. **Nullity/Exception Assertions**

#### Intent Planner
- 输入：`TaskDescription`、动作序列、断言类别（或 `None`）。
- 输出：阶段计划 `phase_plan`，每个阶段包含：
  - `phase_id`
  - `action_range`
  - `intent`
- 规则：
  - 如果 `TaskDescription` 中存在检查关键词，继续沿用静态子句拆分逻辑。
  - 如果**没有** 检查关键词，则不再静态提取 `phase_intent`，改为把**类别名称 + TaskDescription + 动作序列**交给大语言模型划分阶段。

#### Control Selector
- 输入：当前阶段的 `phase_intent`、当前 action 对应的截图与 XML 组件树。
- 输出：与当前阶段最相关的一小组候选组件（供断言生成器使用）。
- 规则：
  - 优先使用静态软匹配评分筛选组件。
  - 如果静态软匹配后**所有组件评分都为 0**，则进入 **LLM 兜底筛选**。
  - LLM 兜底时输入：
    - 当前 action 对应的界面截图
    - 阶段末尾为补充信息而额外向下滑动 3 次后采集到的截图
    - XML 组件列表（已过滤掉明显容器类组件，仅保留 `xpath`、`resource-id`、`text`、`content-desc`、`class`）
    - 当前阶段的 `phase_intent`
  - 要求 LLM：
    - 先分析 `phase_intent`
    - 再基于截图识别目标组件的**文本信息和大致位置**
    - 最后从 XML 候选组件中选出最相关的组件

#### Assertion Generator
- 输入：阶段结束、开始界面截图、候选组件、阶段意图（动作范围）。
- 输出：断言内容。
- 说明：
  - 只负责“生成断言内容”。
  - 不负责组件筛选本身。
  - 当前阶段相关 UI 候选由 `Control Selector` 提供。

#### Assertion Checker
- 输入：Assertion Generator 生成的 assertion。
- 功能：检查断言目标组件是否能在当前界面定位、断言格式是否合规。
- 若可定位：将 assertion 插入到当前 action 后。
- 若不可定位：返回 Assertion Generator 重新生成。

---

## 3. 数据流与流程

### 3.1 总体流程

```text
TaskDescription
      │
      ▼
Assertion Classifier
      │
      ▼
Intent Planner
      │
      ▼
Test Executor 执行 action 并采集截图/XML
      │
      ▼
Control Selector
      │
      ├─ 静态软匹配命中 → 候选组件
      └─ 全部评分为 0 → LLM 兜底筛选
      │
      ▼
Assertion Generator
      │
      ▼
Assertion Checker
      │
      ▼
插入 assertion 到测试用例
```

### 3.2 执行细节

1. `Assertion Classifier` 先判断断言类别。
2. `Intent Planner` 根据任务描述与动作序列生成阶段计划。
3. `Test Executor` 执行测试用例中的每个 action。
4. 每个 action 执行后，生成：
   - 当前截图
   - 当前界面组件树
   - 当前/上一步 action 描述
5. `Control Selector` 先用静态软匹配筛选候选组件。
6. 如果静态评分全部为 0，则使用截图 + XML + `phase_intent` 调用 LLM 兜底筛选。
   - 若当前动作位于阶段末尾，还会把额外下滑 3 次后获得的截图一并提供给 LLM。
7. `Assertion Generator` 基于这些候选组件生成断言。
8. `Assertion Checker` 校验断言目标是否存在。
9. 校验通过后，将断言插入到当前 action 后。
10. 若校验失败，返回 `Assertion Generator` 继续生成。

### 3.3 消融基线说明

当前支持的消融基线包括：

- `--baseline androb2o`：使用 AndroB2O baseline。
- `--ablation no_planner`：关闭阶段规划。
- `--ablation no_selector`：关闭组件筛选。
- `--ablation no_checker`：关闭断言校验。
- `--ablation no_classifier`：关闭断言分类器；此时不再调用 `classifier.py`，后续阶段规划和断言生成仅使用通用断言集合与 `category=None`。

---

## 4. 代码结构说明
这是本机 PC 上的工作框架目录，核心文件如下：

- `classifier.py`：断言类别分类器，输出 1~4 类。
- `intent_planner.py`：阶段规划模块，负责把动作序列拆成 1~2 个逻辑阶段或按检查关键词子句拆分。
- `control_selector.py`：组件筛选模块，负责静态召回与 LLM 兜底筛选。
- `assertion_generator.py`：断言生成模块。
- `assertion_checker.py`：断言校验。
- `assertion_handlers.py`：具体断言函数实现。
- `test_case_runner.py`：测试执行主流程，串联分类、阶段规划、组件筛选、断言生成与校验；支持 `no_classifier` 消融。
- `test_executor.py`：Appium 驱动与测试用例执行入口。
- `baseline_runner.py`：单模型 baseline 运行入口。
- `llm_client.py`：统一管理 OpenAI-compatible 客户端创建与调用。

服务器端目录，保存 Qwen3-VL-32B-Instruct 模型文件与启动脚本：

- `qwen3_vl_openai_server.py`：启动一个 OpenAI-compatible 服务。(请自行创建启动本地模型的代码)

---

## 5. 环境依赖

### 5.1 Python 版本

建议使用：`Python 3.10+`

### 5.2 主要 Python 依赖

PC 端框架常用依赖：

- `openai`
- `appium-python-client`
- `selenium`
- `beautifulsoup4`
- `lxml`
- `requests`
- `pillow`
- `numpy`
- `pyyaml`（如你后续扩展配置）

服务器端模型服务常用依赖：

- `torch`
- `transformers`
- `fastapi`
- `uvicorn`
- `pillow`

> 如果你使用 Qwen3-VL 的多模态能力，建议服务器侧的 `transformers` 版本与模型要求保持一致。

---

## 6. 如何启动服务器端本地模型

在服务器上启动本地模型：

```bash
python qwen3_vl_openai_server.py
```

### 6.1 服务端接口

- Health check: `GET http://<server-ip>:8000/health`
- Chat completions: `POST http://<server-ip>:8000/v1/chat/completions`

### 6.2 说明

这个服务采用 **OpenAI-compatible** 协议，因此 PC 端只需要配置：

- `base_url = http://localhost:8001/v1`
- `api_key = 任意非空字符串`
- `model = Qwen/Qwen3-VL-32B-Instruct`

---

## 7. 如何配置 PC 端框架调用方式

### 7.1 使用服务器上的本地模型

推荐流程如下：

1. 在 PC 端先执行端口转发或 SSH 隧道。
2. 在服务器上启动本地模型服务。
3. 回到 PC 端的本地终端，设置环境变量：

```bash
export ASSERTION_LLM_BASE_URL="http://localhost:8001/v1"
export ASSERTION_LLM_API_KEY="local-dummy-key"
export ASSERTION_LLM_MODEL="Qwen/Qwen3-VL-32B-Instruct"

export BASELINE_LLM_BASE_URL="http://localhost:8001/v1"
export BASELINE_LLM_API_KEY="local-dummy-key"
export BASELINE_LLM_MODEL="Qwen/Qwen3-VL-32B-Instruct"
```

> 注意：本地 OpenAI-compatible 服务通常不校验 API key，但 OpenAI Python SDK 需要一个非空值，所以这里使用占位字符串即可。

4. 在 PC 端运行断言生成框架：

```bash
python test_executor.py input/booking-searchhotel.txt
```

如果你需要关闭断言分类器，可在单文件运行时加上：

```bash
python test_executor.py input/booking-searchhotel.txt --ablation no_classifier
```

批处理脚本也支持同样的开关：

```bash
python run_output_tests.py --no-classifier output
```

### 7.2 使用云端 API

如果你要接云端大模型，只需修改 `testcases/llm_config.json` 中对应条目即可，例如：

```json
{
  "ASSERTION": {
    "base_url": "https://your-openai-compatible-endpoint/v1",
    "api_key": "your-api-key",
    "model": "your-model-name"
  },
  "BASELINE": {
    "base_url": "https://your-openai-compatible-endpoint/v1",
    "api_key": "your-api-key",
    "model": "your-model-name"
  }
}
```

如果你临时需要覆盖配置，也仍然可以使用环境变量；但日常建议只改配置文件。

---

## 8. 如何运行框架

### 8.1 运行断言生成主流程

```bash
python test_executor.py input/booking-searchhotel.txt
```


### 8.2 运行单模型基线生成断言

```bash
python baseline_runner.py input/booking-searchhotel.txt
```

### 8.3 运行 AndroB2O baseline 生成断言

AndroB2O baseline 使用 `test_executor.py` 的独立入口，不经过 `baseline_runner.py`，也不会影响现有消融路径。

```bash
python test_executor.py input/minimal-addtask-and-removetask.txt --baseline androb2o
```

运行后会：

1. 执行输入文件中的每个 action。
2. 每步动作后采集截图和 XML。
3. 调用 `baselines/andro_b2o_adapter.py`，先做元素提取，再做断言生成。
4. 将生成的断言插入到对应 action 后。
5. 输出最终带断言的测试用例到 `baselines/output/<case>/`。

### 8.4 基线与消融模式的关系

- `--baseline none`：默认模式，走你原来的完整断言生成流程。
- `--baseline androb2o`：启用 AndroB2O baseline。

### 8.5 消融实验运行方式

消融实验只针对断言生成主入口 `test_executor.py` 生效，不影响 `run_output_tests.py` 和 `baseline_runner.py`。

`test_executor.py` 支持如下开关：

- `--ablation none`：默认模式，保持完整断言生成流程不变。
- `--ablation no_classifer`：去掉 `classifier.py` 对应的断言分类器。
- `--ablation no_planner`：去掉 `intent_planner.py` 对应的意图规划器。
  - 运行完整测试用例。
  - 最后将所有步骤收集到的 XML 和截图信息合并后，一次性交给组件筛选器，再进行断言生成与检查。
- `--ablation no_selector`：去掉 `control_selector.py` 对应的组件选择器。
  - 不再做组件筛选。
  - 断言生成器直接接收最后一步的 XML 与截图输入。
  - 该消融路径仍保留断言生成与断言检查流程，只是跳过组件筛选器。
- `--ablation no_checker`：去掉 `assertion_checker.py` 对应的断言检查器。
  - 生成出的断言不再做定位检查。
  - 直接插入到测试用例中。

示例：

```bash
# 默认完整流程
python test_executor.py input/booking-searchhotel.txt

# 去掉意图规划器
python test_executor.py input/booking-searchhotel.txt --ablation no_planner

# 去掉组件选择器
python test_executor.py input/booking-searchhotel.txt --ablation no_selector

# 去掉断言检查器
python test_executor.py input/booking-searchhotel.txt --ablation no_checker

# 去掉断言分类器
python test_executor.py input/booking-searchhotel.txt --ablation no_classifier
```
---

## 9. 输出目录说明

- `information/`：截图与中间信息 JSON。
- `output/`：最终生成的带断言测试用例。
- `baseline/`：baseline 生成结果与日志。
- `test/`：执行测试时的运行工件。

---

## 10. 数据集说明：AppTask-Assert Set

为了便于在本仓库中统一管理断言生成任务，我把这套数据集暂命名为 **AppTask-Assert Set**。

### 10.1 数据集组成

数据集分为两个部分：

- `input/`：**输入任务集**。每个 `.txt` 文件包含：
  - `appPackage`
  - `appActivity`
  - `TaskDescription`
  - 动作序列（action plan）
- `groundtruth/`：**标准答案集**。按应用名划分目录，用于存放与对应任务匹配的人工标注结果或参考断言。

当前统计如下：

- `input/` 中共有 **58** 个任务文件
- `groundtruth/` 中共有 **25** 个应用目录

### 10.2 目录组织方式

`input/` 中的文件名通常采用：

```text
<app>-<task>.txt
```

例如：

- `booking-searchhotel.txt`
- `calculator-bmi.txt`
- `yelp-share-restaurant.txt`

`groundtruth/` 中则按应用维度组织，例如：

- `booking.com/`
- `calculator/`
- `youtube/`

### 10.3 数据特点

AppTask-Assert Set 覆盖了多种常见的移动端断言场景，包括但不限于：

- 搜索与跳转
- 登录/注册异常提示
- 主题切换与设置项检查
- 分享到其他应用
- 列表新增/删除/保存
- 计算类结果校验

这使它既适合验证断言生成能力，也适合评估组件筛选、阶段规划和断言校验的整体效果。

---

## 11. 设计上的约束与说明

1. **断言类别分类器只负责分类**：输出类别 1~4，不直接生成断言。
2. **阶段规划区分两条路径**：
   - 含检查关键词：可按检查子句拆阶段。
   - 不含检查关键词：由 LLM 结合类别名、TaskDescription、动作序列划分 1~3 个阶段。
3. **组件筛选层优先静态召回**：只有静态软匹配全 0 时才使用 LLM 兜底。
4. **LLM 兜底必须先看截图**：筛组件时要先分析截图中的组件文本与大致位置，再结合 XML 候选组件做选择。
5. **Assertion Generator 不承担筛选职责**：它只消费 `Control Selector` 给出的候选组件并生成断言。
6. **baseline 逻辑不变**：仅替换模型 client 初始化与调用封装。
7. **本机 PC 与服务器分离**：`testcases` 在 PC 端，`model` 在服务器端。
8. **兼容两种 LLM 接入**：API-key 云端接入与服务器本地模型接入。

---

## 12. 推荐检查项

在正式运行前，建议确认：

- Appium 服务已启动。
- PC 可访问服务器的 `base_url`。
- 服务器端模型服务已成功加载本地模型。
- `openai` Python 包已安装。
- 如果使用本地服务，`PTIDroid/llm_config.json` 中的地址、模型名与 API key 需要配置正确。
