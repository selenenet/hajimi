# 🚀 HAJIMI Gemini API Proxy

这是一个基于 FastAPI 构建的 Gemini API 代理，旨在提供一个简单、安全且可配置的方式来访问 Google 的 Gemini 模型。适用于在 Hugging Face Spaces 上部署，并支持openai api格式的工具集成。

## 项目采用动态更新，随时会有一些小更新同步到主仓库且会自动构建镜像，如果反馈的bug开发者说修了但是版本号没变是正常现象，~~记得勤更新镜像哦~~

## 错误自查

遇到问题请先查看以下的 **错误自查** 文档，确保已尝试按照其上的指示进行了相应的排查与处理。

- [错误自查](./wiki/error.md)

###  使用文档

- [Claw Cloud部署的使用文档（免费，手机电脑均可使用）](./wiki/claw.md) 感谢[@IDeposit](https://github.com/IDeposit)编写

- [huggingface 部署的使用文档（免费，手机电脑均可使用）](./wiki/huggingface2.md)

- [termux部署的使用文档（手机使用）](./wiki/Termux.md) 感谢 [@天命不又](https://github.com/tmby) 编写

- [Render 部署的使用文档 (免费，手机电脑均可使用，但可能需要绑信用卡)](./wiki/render.md) 感谢 [@LSR](https://github.com/lesser0) 编写

- [windows 本地部署的使用文档](./wiki/windows.md)

- [vertex模式的使用文档](./wiki/vertex.md)

- ~~[zeabur部署的使用文档(需付费)](./wiki/zeabur.md) 感谢**墨舞ink**编写~~（已过时且暂时无人更新，欢迎提交pull requests）

###  更新日志
* v1.0.2
   * 修复 400 错误

* v1.0.1
   * 新增`清除失效密钥`功能
   * 新增`输出有效秘钥`功能

## ✨ 主要功能：

### 🔑 API 密钥轮询和管理

### 📑 模型列表接口

### 💬 聊天补全接口：

*   提供 `/v1/chat/completions` 接口，支持流式和非流式响应，支持函数调用，与 OpenAI API 格式兼容。
*   支持的输入内容: 文本、文件、图像
*   自动将 OpenAI 格式的请求转换为 Gemini 格式。

### 🔒 密码保护（可选）：

*   通过 `PASSWORD` 环境变量设置密码。
*   提供默认密码 `"123"`。

### 🧩 服务兼容

*   提供的接口与 OpenAI API 格式兼容,便于接入各种服务

### ⚙️ 功能配置

* 方式 1 : 通过网页前端进行配置
* 方式 2 : 根据 [配置文档](./app/config/settings.py) 中的注释说明，修改对应的变量

## ⚠️ 注意事项：

*   **强烈建议在生产环境中设置 `PASSWORD` 环境变量，并使用强密码。**
*   根据你的使用情况调整速率限制相关的环境变量。
*   确保你的 Gemini API 密钥具有足够的配额。

## 💡 特色功能：

### 🎭 假流式传输

*   **作用：** 解决部分网络环境下客户端通过非流式请求 Gemini 时可能遇到的断连问题。**默认开启**。

*   **原理简述：** 当客户端请求流式响应时，本代理会每隔一段时间向客户端发出一个空信息以维持连接，同时在后台向 Gemini 发起一个完整的、非流式的请求。等 Gemini 返回完整响应后，再一次性将响应发回给客户端。

*   **注意：** 如果想使用真的流式请求，请**关闭**该功能

### ⚡ 并发与缓存

*   **作用：** 允许您为用户的单次提问同时向 Gemini 发送多个请求，并将额外的成功响应缓存起来，用于后续重新生成回复。

*   **注意：** 此功能**默认关闭** 。只有当您将并发数设置为 2 或以上时，缓存才会生效。缓存匹配要求提问的上下文与被缓存的问题**完全一致**（包括标点符号）。此外，该模式目前仅支持非流式及假流式传输
    
    **Q: 新版本增加的并发缓存功能会增加 gemini 配额的使用量吗？**
   
    **A: 不会**。因为默认情况下该功能是关闭的。只有当你主动将并发数 `CONCURRENT_REQUESTS` 设置为大于 1 的数值时，才会实际发起并发请求，这才会消耗更多配额。
   
    **Q: 如何使用并发缓存功能？**
   
    **A:** 修改并发请求数，使其等于你想在一次用户提问中同时向 Gemini 发送的请求数量（例如设置为 `3`）。
    
    这样设置后，如果一次并发请求中收到了多个成功的响应，除了第一个返回给用户外，其他的就会被缓存起来。

### 🎭 伪装信息

*   **作用：** 在发送给 Gemini 的消息中添加一段随机生成的、无意义的字符串，用于"伪装"请求，可能有助于防止被识别为自动化程序。**默认开启**。

*   **注意：** 如果使用非 SillyTavern 的其余客户端 (例如 cherryStudio )，请**关闭**该功能

### 🌐 联网模式

*   **作用：** 让 Gemini 模型能够利用搜索工具进行联网搜索，以回答需要最新信息或超出其知识库范围的问题。

*   **如何使用：**

    在客户端请求时，选择模型名称带有 `-search` 后缀的模型（例如 `gemini-2.5-pro-search`，具体可用模型请通过 `/v1/models` 接口查询）。

### 🚦 速率限制和防滥用：

*   通过环境变量自定义限制：
    *   `MAX_REQUESTS_PER_MINUTE`：每分钟最大请求数（默认 30）。
    *   `MAX_REQUESTS_PER_DAY_PER_IP`：每天每个 IP 最大请求数（默认 600）。
*   超过速率限制时返回 429 错误。

## 📡 API 接口说明

### 模型列表接口

- **GET** `/v1/models` - 获取可用模型列表
- **GET** `/models` - 获取可用模型列表（兼容旧版本）

### 聊天补全接口

- **POST** `/v1/chat/completions` - 聊天补全接口，支持流式和非流式响应
- **POST** `/chat/completions` - 聊天补全接口（兼容旧版本）

### Nano Banana 图片接口（Vertex 模式）

- **POST** `/v1/images/generations` - OpenAI 格式文生图接口
- **POST** `/v1/images/edits` - OpenAI multipart 格式多图编辑接口
- **POST** `/gemini/v1/models/{image-model}:generateContent` - Gemini 原生图片接口
- **POST** `/gemini/v1/models/{image-model}:streamGenerateContent` - Gemini 原生 SSE 图片接口

当前图片模型：`gemini-3.1-flash-image`、`gemini-3.1-flash-lite-image`、
`gemini-3-pro-image` 和兼容旧版 `gemini-2.5-flash-image`。OpenAI 接口的模型
ID 带 `[PAY]` 前缀，默认使用 `[PAY]gemini-3.1-flash-image`。

文生图示例：

```bash
curl https://example.com/v1/images/generations \
  -H "Authorization: Bearer $HAJIMI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "[PAY]gemini-3.1-flash-image",
    "prompt": "A blue paper crane on a clean white desk",
    "size": "1024x1024",
    "quality": "low",
    "output_format": "webp",
    "output_compression": 80,
    "response_format": "b64_json"
  }'
```

图片编辑示例：

```bash
curl https://example.com/v1/images/edits \
  -H "Authorization: Bearer $HAJIMI_API_KEY" \
  -F "model=[PAY]gemini-3.1-flash-image" \
  -F "prompt=Add a small knitted wizard hat" \
  -F "image=@cat.png;type=image/png" \
  -F "quality=medium" \
  -F "output_format=png" \
  -F "response_format=b64_json"
```

OpenAI 图片接口只返回 `b64_json`，不会为图片创建公开 URL 或持久化文件。
`n` 当前必须为 `1`，不支持 `stream`、`partial_images` 或 `mask`。编辑接口同时
接受重复的 `image` 和 `image[]` 字段。

- `size` 支持 `auto`、`1024x1024`、`1536x1024` 和 `1024x1536`。
- `quality` 映射为输出分辨率：`low=512`、`medium/standard=1K`、
  `high/hd=2K`。目标模型不支持该分辨率时返回 400，不自动降级。
- `output_format` 支持 `png`、`jpeg`、`webp`；`output_compression` 为
  `0-100`，仅适用于 JPEG/WebP。
- `background` 支持 `auto`、`transparent`、`opaque`，`style` 支持
  `vivid`、`natural`；这些值通过提示词近似实现，不是 Gemini 原生保证。
- `moderation=auto/low` 均保留 Google 自身安全策略，不会降低上游过滤级别。
- 扩展字段 `aspect_ratio` 优先于 `size`；`image_size` 优先于 `quality`。

Gemini 原生接口可直接传递 `generationConfig.imageConfig`，适合使用 OpenAI
字段未覆盖的 Gemini 高级选项。

### Vertex 模式接口

- **GET** `/vertex/models` - 获取 Vertex 模型列表
- **POST** `/vertex/chat/completions` - Vertex 聊天补全接口

### AI Studio 模式接口

- **GET** `/aistudio/models` - 获取 AI Studio 模型列表
- **POST** `/aistudio/chat/completions` - AI Studio 聊天补全接口

## ⚙️ 配置选项

项目支持多种配置方式，可以通过环境变量进行配置：

### 基础配置

- `PASSWORD` - API访问密码，默认为"123"
- `WEB_PASSWORD` - 网页配置密码，默认与PASSWORD相同
- `GEMINI_API_KEYS` - Gemini API密钥，多个密钥用逗号分隔
- `FAKE_STREAMING` - 是否启用假流式传输，默认为true
- `STORAGE_DIR` - 配置持久化存储目录
- `ENABLE_STORAGE` - 是否启用配置持久化存储，默认为false

### 并发与缓存配置

- `CONCURRENT_REQUESTS` - 并发请求数，默认为1
- `INCREASE_CONCURRENT_ON_FAILURE` - 失败时增加的并发数，默认为0
- `MAX_CONCURRENT_REQUESTS` - 最大并发请求数，默认为3
- `CACHE_EXPIRY_TIME` - 缓存过期时间（秒），默认为21600（6小时）
- `MAX_CACHE_ENTRIES` - 最大缓存条目数，默认为500
- `CALCULATE_CACHE_ENTRIES` - 计算缓存键时使用的消息数量，默认为6
- `PRECISE_CACHE` - 是否使用精确缓存，默认为false

### Vertex AI 配置

- `ENABLE_VERTEX` - 是否启用Vertex AI，默认为false
- `GOOGLE_CREDENTIALS_JSON` - Google凭证JSON
- `ENABLE_VERTEX_EXPRESS` - 是否启用Vertex Express模式，默认为false
- `VERTEX_EXPRESS_API_KEY` - Vertex Express API密钥

### 联网搜索配置

- `SEARCH_MODE` - 是否启用联网搜索模式，默认为false
- `SEARCH_PROMPT` - 联网搜索提示语

### 安全配置

- `RANDOM_STRING` - 是否启用随机字符串伪装，默认为true
- `RANDOM_STRING_LENGTH` - 随机字符串长度，默认为5
- `MAX_EMPTY_RESPONSES` - 最大空响应重试次数，默认为5

### 速率限制配置

- `MAX_RETRY_NUM` - 请求时的最大总轮询key数，默认为15
- `MAX_REQUESTS_PER_MINUTE` - 每分钟最大请求数，默认为30
- `MAX_REQUESTS_PER_DAY_PER_IP` - 每天每个IP最大请求数，默认为600
- `API_KEY_DAILY_LIMIT` - 每个API密钥每天使用限制，默认为100

### 模型过滤配置

- `BLOCKED_MODELS` - 屏蔽的模型列表，用逗号分隔
- `WHITELIST_MODELS` - 白名单模型列表，用逗号分隔
- `WHITELIST_USER_AGENT` - 白名单User-Agent列表，用逗号分隔

### 其他配置

- `PUBLIC_MODE` - 公益站模式，默认为false
- `DASHBOARD_URL` - 前端地址
- `ALLOWED_ORIGINS` - 允许的跨域源列表，用逗号分隔

## 🚀 部署方式

### Hugging Face Spaces 部署

1. Fork本项目到你的GitHub账户
2. 在GitHub Actions中构建Docker镜像
3. 在Hugging Face Spaces中创建新的Docker空间
4. 配置环境变量（PASSWORD和GEMINI_API_KEYS为必需）
5. 部署并访问前端界面

### Claw Cloud 部署

1. 注册Claw Cloud账户
2. 登录Claw Cloud控制台
3. 进入APP Launchpad并创建新应用
4. 使用镜像地址 `ghcr.io/wyeeeee/hajimi:latest`
5. 配置环境变量
6. 部署应用并获取访问地址

### Vertex AI 部署

1. 在Google Cloud Platform中创建服务账户
2. 获取服务账户的JSON凭证文件
3. 在项目中启用Vertex AI API
4. 配置环境变量 `ENABLE_VERTEX=true` 和 `GOOGLE_CREDENTIALS_JSON`
5. 部署应用

### 本地部署

1. 克隆项目代码
2. 安装依赖：`pip install -r requirements.txt`
3. 配置环境变量
4. 运行应用：`uvicorn app.main:app --host 0.0.0.0 --port 7860`

## 📚 更多文档

- [配置文档](./app/config/settings.py) - 详细的配置选项说明
- [错误自查](./wiki/error.md) - 常见问题和解决方案
- [Hugging Face部署指南](./wiki/huggingface2.md) - 详细的Hugging Face部署教程
- [Claw Cloud部署指南](./wiki/claw.md) - 详细的Claw Cloud部署教程
- [Vertex模式指南](./wiki/vertex.md) - Vertex AI配置和使用指南
