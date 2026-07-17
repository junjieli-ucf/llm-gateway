"""Day 6 周项目:llm-gateway —— Week 1 的整合收官作品。

把整周学的东西拧成一个真实服务:
  - 类型标注 + Pydantic 校验(Day1)
  - async 端点 + await 调用外部 API(Day2)
  - 可被 pytest + mock 测试(Day3,见 tests/test_main.py)
  - FastAPI 路由 / 请求体 / 依赖注入 / HTTPException(Day4)
  - 请求日志 middleware + lifespan(Day5)
  - 调用 Claude API 返回结果(今天新学,最简调用)

功能:POST /v1/complete 收一个 prompt → 调 Claude → 返回生成文本。

运行(先装依赖):
    uv add fastapi "uvicorn[standard]" anthropic
    uv add --dev pytest pytest-asyncio ruff mypy

    # 需要一个 API key(环境变量):
    export ANTHROPIC_API_KEY=sk-...            # Claude 官方
    # 或用 Z.ai/GLM 兼容端点(Anthropic 兼容,改 base_url + key 即可):
    # export ANTHROPIC_BASE_URL=https://...    # SDK 会自动读这个环境变量
    export GATEWAY_API_KEY=my-secret           # 本网关自己的鉴权 key

    uv run uvicorn main:app --reload

测试:  http://127.0.0.1:8000/docs
类型检查:uv run mypy --strict main.py
"""

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import anthropic
from anthropic import AsyncAnthropic
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

# ============================================================
# 配置(从环境变量读,不写死在代码里)
# ============================================================

DEFAULT_MODEL = "claude-opus-4-8"                 # 官方 API 参考推荐的默认模型
# 本网关自己的鉴权 key(注意:这跟 Claude 的 ANTHROPIC_API_KEY 是两回事)
GATEWAY_API_KEY = os.environ.get("GATEWAY_API_KEY", "dev-secret")


# ============================================================
# lifespan:启动时建一个【共享的】Claude 客户端,全程复用(Day5)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动时建 AsyncAnthropic 客户端一次,挂到 app.state 给所有请求复用。

    为什么用 lifespan 建:建客户端有开销,不该每个请求都新建一个。
    AsyncAnthropic() 会自动从环境变量读 ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL。
    """
    print("[lifespan] 启动:创建共享 Claude 客户端")
    app.state.llm = AsyncAnthropic()              # 共享资源:见 get_llm 依赖
    yield
    print("[lifespan] 关闭:清理客户端")
    await app.state.llm.close()


app = FastAPI(title="llm-gateway", lifespan=lifespan)


# ============================================================
# middleware:请求日志(Day5)——每个请求打印方法/路径/耗时
# ============================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    print(f"[req] {request.method} {request.url.path} -> {response.status_code} ({elapsed:.3f}s)")
    return response


# ============================================================
# 依赖:鉴权 + 取共享客户端(Day4 的 Depends)
# ============================================================

def verify_api_key(x_api_key: str = Header()) -> str:
    """从请求头 X-API-Key 校验本网关的访问凭证。key 错 -> 401。"""
    if x_api_key != GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


def get_llm(request: Request) -> AsyncAnthropic:
    """把 lifespan 建好的共享客户端取出来。

    单独抽成依赖有个好处:测试时可以用 app.dependency_overrides 换成假客户端,
    从而【不真的调 Claude】(见测试文件)——这就是 Day3 学的 mock 思想。
    """
    # app.state 是无类型的(属性访问返回 Any),用 cast 明确告诉 mypy 它是什么。
    return cast(AsyncAnthropic, request.app.state.llm)


# ============================================================
# Pydantic 模型:请求体校验 + 响应结构(Day1 + Day4)
# ============================================================

class CompleteRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=10000, description="要发给模型的提示词")
    max_tokens: int = Field(default=1024, gt=0, le=8192, description="最多生成多少 token")
    model: str = Field(default=DEFAULT_MODEL, description="用哪个模型")


class CompleteResponse(BaseModel):
    text: str                    # 模型生成的文本
    model: str                   # 实际使用的模型
    input_tokens: int            # 输入 token 数(计费用)
    output_tokens: int           # 输出 token 数


# ============================================================
# 端点:POST /v1/complete —— 收 prompt,调 Claude,返回结果
# ============================================================

@app.post("/v1/complete", response_model=CompleteResponse)
async def complete(
    req: CompleteRequest,                              # 请求体自动校验(Day1/Day4)
    api_key: str = Depends(verify_api_key),            # 先过鉴权(Day4)
    llm: AsyncAnthropic = Depends(get_llm),            # 注入共享客户端
) -> CompleteResponse:
    """async 端点:await 调用 Claude 期间,服务器可处理别的请求(Day2)。"""
    try:
        # 最简调用:给 model / max_tokens / messages 即可(官方 API 参考)
        message = await llm.messages.create(
            model=req.model,
            max_tokens=req.max_tokens,
            messages=[{"role": "user", "content": req.prompt}],
        )
    except anthropic.APIStatusError as e:              # Claude 返回了错误状态码
        # 把上游错误转成网关的 502(Bad Gateway):不是我们的错,是上游的
        raise HTTPException(status_code=502, detail=f"LLM error: {e.message}") from e
    except anthropic.APIConnectionError as e:          # 连不上 Claude
        raise HTTPException(status_code=503, detail="LLM unavailable") from e

    # 响应 content 是一个 block 列表,把其中的文本拼起来(用 getattr 对 mypy 友好)
    text = "".join(getattr(b, "text", "") for b in message.content)
    return CompleteResponse(
        text=text,
        model=message.model,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
    )


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """健康检查端点:CI / 部署时用来确认服务活着(无需鉴权)。"""
    return {"status": "ok"}
