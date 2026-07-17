"""Day 6 测试:给 llm-gateway 写单元测试,【mock 掉真实的 Claude 调用】。

核心思想(Day3):单元测试不能真的联网、真的花 API 钱。
所以我们用一个"假客户端"替换掉真的 Claude 客户端,让它立刻返回预设结果。
这样测试又快、又稳、又免费,还能精确制造各种场景(成功/失败)。

运行:  uv run pytest -v
"""

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import main as gw
from main import app, get_llm

VALID_KEY = gw.GATEWAY_API_KEY   # 测试用的合法网关 key


# ============================================================
# 造一个"假 Claude 客户端"
# ============================================================

def _fake_message(text: str) -> SimpleNamespace:
    """伪造一个 Claude 响应对象,长得跟真的 message 一样(有我们代码要用的字段)。"""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],   # content 是 block 列表
        model="claude-opus-4-8",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


@pytest.fixture
def client_with_fake_llm() -> Iterator[TestClient]:
    """fixture(Day3):准备一个 TestClient,并把真客户端换成假的。

    app.dependency_overrides 是 FastAPI 的官方测试机制:
    覆盖掉 get_llm 依赖,让端点拿到我们的假客户端,而不是真的 AsyncAnthropic。
    """
    fake_llm = SimpleNamespace(messages=SimpleNamespace())
    # 假的 messages.create 是个 AsyncMock(异步函数的替身),默认返回一条假消息
    fake_llm.messages.create = AsyncMock(return_value=_fake_message("你好,我是假回复"))

    app.dependency_overrides[get_llm] = lambda: fake_llm   # 关键:注入假客户端
    with TestClient(app) as client:                        # TestClient 会跑 lifespan
        client.fake_llm = fake_llm                          # 挂上去方便测试里断言调用
        yield client
    app.dependency_overrides.clear()                       # 清理,避免污染其他测试


# ============================================================
# 测试用例(≥ 覆盖:成功、鉴权失败、校验失败、上游错误、健康检查)
# ============================================================

def test_healthz(client_with_fake_llm: TestClient) -> None:
    """健康检查应返回 200 + ok。"""
    resp = client_with_fake_llm.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_complete_success(client_with_fake_llm: TestClient) -> None:
    """正常路径:带对的 key + 合法请求体 -> 200,返回假客户端预设的文本。"""
    resp = client_with_fake_llm.post(
        "/v1/complete",
        headers={"X-API-Key": VALID_KEY},
        json={"prompt": "讲个笑话", "max_tokens": 50},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "你好,我是假回复"        # 来自假客户端,证明没真的调 Claude
    assert data["model"] == "claude-opus-4-8"
    assert data["input_tokens"] == 10


def test_complete_calls_llm_once(client_with_fake_llm: TestClient) -> None:
    """断言真的调用了(一次)LLM,且参数正确 —— 用 mock 的 call_count / 入参。"""
    client_with_fake_llm.post(
        "/v1/complete",
        headers={"X-API-Key": VALID_KEY},
        json={"prompt": "hi"},
    )
    create = client_with_fake_llm.fake_llm.messages.create   # type: ignore[attr-defined]
    assert create.call_count == 1
    # 检查传给 Claude 的参数:prompt 被正确放进 messages
    kwargs = create.call_args.kwargs
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_missing_api_key_rejected(client_with_fake_llm: TestClient) -> None:
    """不带 X-API-Key 头 -> 422(缺必填头);Claude 不应被调用。"""
    resp = client_with_fake_llm.post("/v1/complete", json={"prompt": "hi"})
    assert resp.status_code == 422
    assert client_with_fake_llm.fake_llm.messages.create.call_count == 0  # type: ignore[attr-defined]


def test_wrong_api_key_rejected(client_with_fake_llm: TestClient) -> None:
    """带错的 key -> 401;端点逻辑不执行。"""
    resp = client_with_fake_llm.post(
        "/v1/complete",
        headers={"X-API-Key": "wrong"},
        json={"prompt": "hi"},
    )
    assert resp.status_code == 401


@pytest.mark.parametrize(
    "bad_body",
    [
        {"prompt": ""},                      # prompt 太短(min_length=1)
        {"prompt": "hi", "max_tokens": 0},   # max_tokens 必须 > 0
        {"prompt": "hi", "max_tokens": 99999},  # 超过上限 8192
        {},                                  # 缺 prompt
    ],
)
def test_invalid_body_rejected(client_with_fake_llm: TestClient, bad_body: dict[str, object]) -> None:
    """parametrize(Day3):多组脏请求体,都应被 Pydantic 自动 422 拦下。"""
    resp = client_with_fake_llm.post(
        "/v1/complete",
        headers={"X-API-Key": VALID_KEY},
        json=bad_body,
    )
    assert resp.status_code == 422


def test_upstream_error_becomes_502(client_with_fake_llm: TestClient) -> None:
    """用 side_effect 让假客户端抛 Claude 的 APIStatusError,断言网关转成 502。"""
    import anthropic
    import httpx

    # 造一个"真的" httpx 请求/响应,APIStatusError 内部需要用到 response.request
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(500, request=request)
    err = anthropic.APIStatusError("boom", response=response, body=None)
    client_with_fake_llm.fake_llm.messages.create = AsyncMock(side_effect=err)  # type: ignore[attr-defined]

    resp = client_with_fake_llm.post(
        "/v1/complete",
        headers={"X-API-Key": VALID_KEY},
        json={"prompt": "hi"},
    )
    assert resp.status_code == 502   # 上游错 -> 网关回 502,而不是自己崩掉
