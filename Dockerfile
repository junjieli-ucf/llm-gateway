# llm-gateway 的 Dockerfile
# 作用:把"代码 + 依赖 + Python 环境"打包成一个标准镜像,任何装了 Docker 的机器都能一样地跑。
#
# 构建:  docker build -t llm-gateway .
# 运行:  docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... -e GATEWAY_API_KEY=my-secret llm-gateway

# --- 1. 基础镜像:一个已装好 Python 3.12 的精简系统 ---
# slim = 精简版,体积小、只含必要东西,适合生产。
FROM python:3.12-slim

# --- 2. 容器内的工作目录(相当于 cd /app),后续命令都在这里执行 ---
WORKDIR /app

# --- 3. 层缓存优化:先只复制"依赖清单",不复制代码 ---
# 为什么先复制这两个?因为依赖不常变,而代码常改。
# 这样只要依赖没变,下面"装依赖"那一层就用缓存,改代码时不用重装依赖 → 构建飞快。
COPY pyproject.toml uv.lock ./

# --- 4. 装 uv,再用它安装依赖 ---
# --frozen:严格按 uv.lock 装,保证环境可复现,不擅自升级。
RUN pip install uv && uv sync --frozen

# --- 5. 现在才复制其余代码(常变的东西放后面,不影响上面依赖层的缓存) ---
COPY . .

# --- 6. 安全:创建一个非 root 用户并切换过去 ---
# 不用最高权限跑应用,万一被攻破,攻击者拿到的也只是受限用户,危害小。
RUN useradd --create-home appuser
USER appuser

# --- 7. 声明容器对外用 8000 端口(文档性质,真正暴露还要 docker run -p) ---
EXPOSE 8000

# --- 8. 容器启动时运行的命令 ---
# 关键:--host 0.0.0.0(不是 127.0.0.1!)。
# 容器里 127.0.0.1 只指容器自己内部,外面访问不到;0.0.0.0 表示"接受任何来源",
# 配合 docker run -p 才能从你电脑访问到容器里的服务。
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
