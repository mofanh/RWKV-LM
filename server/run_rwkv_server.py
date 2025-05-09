from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import types, torch, copy, time, gc, os, logging
import numpy as np
import uvicorn
from typing import List, Dict, Optional, Any, Union, AsyncGenerator
import asyncio
import json
import sys

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rwkv_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rwkv-server")

# 导入RWKV模型相关代码 - 使用自定义模块
# 设置环境变量(可选)
os.environ["RWKV_PATH"] = "/root/megrez-tmp/RWKV-LM/RWKV-v7"

# 添加模块路径
module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rwkv_module")
sys.path.append(module_dir)

# 导入模块
try:
    logger.info("正在导入RWKV模型组件...")
    from rwkv_module import RWKV_TOKENIZER, RWKV_x070, sample_logits
    logger.info("RWKV模型组件导入成功")
except Exception as e:
    logger.error(f"导入模块失败: {str(e)}")
    raise

# 配置模型参数
args = types.SimpleNamespace()
args.MODEL_NAME = "/root/megrez-tmp/models/RWKV-x070-World-2.9B-v3-20250211-ctx4096"
args.n_layer = 32
args.n_embd = 2560
args.vocab_size = 65536
args.head_size = 64

# 创建FastAPI应用
app = FastAPI(title="RWKV-7 API服务", description="基于RWKV-7模型的文本生成服务")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
model = None
tokenizer = None
model_states = {}

# 请求和响应模型
class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = Field(default=100, ge=1, le=1000)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    top_p: float = Field(default=0.7, ge=0.0, le=1.0)
    session_id: Optional[str] = None
    stream: bool = False
    isDialog: bool = Field(default=False, description="如果为true，则在检测到'\\n\\n'时停止生成")

class GenerationResponse(BaseModel):
    generated_text: str
    session_id: str
    execution_time: float

class StreamResponse(BaseModel):
    token: str
    session_id: str
    finished: bool = False

class SessionRequest(BaseModel):
    session_id: str
    
class SessionsResponse(BaseModel):
    sessions: List[str]

class ServerStatusResponse(BaseModel):
    status: str
    model_name: str
    n_layer: int
    n_embd: int
    active_sessions: int
    uptime: float
    memory_usage: Dict[str, float]

# 服务器启动时间
start_time = time.time()

# 初始化模型
def load_model():
    global model, tokenizer
    
    logger.info("加载RWKV-7模型...")
    try:
        model = RWKV_x070(args)
        tokenizer = RWKV_TOKENIZER("/root/megrez-tmp/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt")
        logger.info(f"模型 {args.MODEL_NAME} 加载完成！")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    load_model()
    logger.info("API服务已启动")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("API服务正在关闭")
    clear_memory()

def get_memory_usage():
    """获取内存使用情况"""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    return {"allocated_gb": allocated, "reserved_gb": reserved}

# API端点
@app.get("/")
async def root():
    return {"message": "RWKV-7 API服务已启动，请使用 /generate 端点进行文本生成"}

@app.get("/status", response_model=ServerStatusResponse)
async def server_status():
    """获取服务器状态"""
    return {
        "status": "运行中" if model is not None else "模型未加载",
        "model_name": os.path.basename(args.MODEL_NAME),
        "n_layer": args.n_layer,
        "n_embd": args.n_embd,
        "active_sessions": len(model_states),
        "uptime": time.time() - start_time,
        "memory_usage": get_memory_usage()
    }

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    """生成文本，支持普通和流式输出"""
    global model, tokenizer, model_states
    
    if not model or not tokenizer:
        logger.error("模型尚未加载")
        raise HTTPException(status_code=503, detail="模型尚未加载")
    
    start_time = time.perf_counter()
    prompt = request.prompt
    logger.info(f"收到生成请求: prompt_length={len(prompt)}, max_length={request.max_length}, stream={request.stream}")
    
    # 生成会话ID或使用现有会话
    session_id = request.session_id if request.session_id else f"session_{int(time.time())}"
    
    # 获取或初始化状态
    if session_id in model_states:
        logger.info(f"使用现有会话: {session_id}")
        init_out, init_state = model_states[session_id]
    else:
        logger.info(f"创建新会话: {session_id}")
        try:
            tokens = tokenizer.encode(prompt)
            init_out, init_state = model.forward(tokens, None)
            model_states[session_id] = (init_out, copy.deepcopy(init_state))
            
            if not request.stream:
                return GenerationResponse(
                    generated_text=prompt,
                    session_id=session_id,
                    execution_time=time.perf_counter() - start_time
                )
            else:
                # 对于流式输出，我们返回提示词
                async def stream_init_response():
                    yield json.dumps({"token": prompt, "session_id": session_id, "finished": False})
                    yield json.dumps({"token": "", "session_id": session_id, "finished": True})
                
                return StreamingResponse(stream_init_response(), media_type="application/json")
        except Exception as e:
            logger.error(f"初始化会话失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"初始化会话失败: {str(e)}")
    
    # 处理流式输出
    if request.stream:
        return StreamingResponse(
            generate_stream(
                init_out, 
                init_state, 
                session_id, 
                request.max_length, 
                request.temperature, 
                request.top_p,
                request.isDialog
            ),
            media_type="application/json"
        )
    
    # 处理常规输出
    return await generate_normal(
        init_out, 
        init_state, 
        session_id, 
        request.max_length, 
        request.temperature, 
        request.top_p, 
        start_time,
        request.isDialog
    )

async def generate_normal(
    init_out, 
    init_state, 
    session_id, 
    max_length, 
    temperature, 
    top_p, 
    start_time,
    isDialog=False
):
    """常规文本生成（非流式）"""
    try:
        out, state = init_out.clone(), copy.deepcopy(init_state)
        all_tokens = []
        generated_text = ""
        
        for i in range(max_length):
            token = sample_logits(out, temperature, top_p)
            all_tokens.append(token)
            
            # 每生成几个token，检查是否出现了"\n\n"
            if isDialog and len(all_tokens) >= 2:
                current_text = tokenizer.decode(all_tokens)
                if "\n\n" in current_text:
                    # 如果找到 "\n\n"，截断到该位置
                    end_pos = current_text.find("\n\n") + 2  # 包含"\n\n"
                    generated_text = current_text[:end_pos]
                    # 注意：这里可能需要重新编码和解码，以确保token边界对齐
                    break
            
            out, state = model.forward(token, state)
        
        # 如果没有提前结束，解码所有生成的token
        if not generated_text:
            generated_text = tokenizer.decode(all_tokens)
        
        # 更新会话状态
        model_states[session_id] = (out, copy.deepcopy(state))
        
        logger.info(f"完成生成: session={session_id}, tokens={len(all_tokens)}")
        
        return GenerationResponse(
            generated_text=generated_text,
            session_id=session_id,
            execution_time=time.perf_counter() - start_time
        )
    except Exception as e:
        logger.error(f"生成过程中出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成过程中出错: {str(e)}")
    
async def generate_stream(
    init_out, 
    init_state, 
    session_id, 
    max_length, 
    temperature, 
    top_p,
    isDialog=False
):
    """流式文本生成"""
    try:
        out, state = init_out.clone(), copy.deepcopy(init_state)
        out_last = 0
        all_tokens = []
        accumulated_text = ""
        
        for i in range(max_length):
            token = sample_logits(out, temperature, top_p)
            all_tokens.append(token)
            
            # 尝试解码当前token
            try:
                tmp = tokenizer.decode(all_tokens[out_last:])
                if "\ufffd" not in tmp:  # 确保是有效的UTF-8字符串
                    yield json.dumps({"token": tmp, "session_id": session_id, "finished": False})
                    accumulated_text += tmp
                    out_last = i + 1
                    
                    # 检查是否为对话模式且出现了"\n\n"
                    if isDialog and "\n\n" in accumulated_text:
                        # 更新会话状态
                        model_states[session_id] = (out, copy.deepcopy(state))
                        # 发送完成信号
                        yield json.dumps({"token": "", "session_id": session_id, "finished": True})
                        logger.info(f"对话模式流式生成已完成: session={session_id}, tokens={len(all_tokens)}")
                        return
            except:
                # 如果解码失败，继续尝试后续token
                pass
            
            # 生成下一个token
            out, state = model.forward(token, state)
            
            # 适当的速率控制，防止服务器过载
            await asyncio.sleep(0.01)
        
        # 确保所有剩余tokens都被解码
        if out_last < len(all_tokens):
            try:
                tmp = tokenizer.decode(all_tokens[out_last:])
                if tmp:
                    yield json.dumps({"token": tmp, "session_id": session_id, "finished": False})
            except:
                pass
        
        # 更新会话状态
        model_states[session_id] = (out, copy.deepcopy(state))
        
        # 发送完成信号
        yield json.dumps({"token": "", "session_id": session_id, "finished": True})
        
        logger.info(f"流式生成完成: session={session_id}, tokens={len(all_tokens)}")
    
    except Exception as e:
        logger.error(f"流式生成过程中出错: {str(e)}")
        # 发送错误信号
        yield json.dumps({"token": f"[ERROR: {str(e)}]", "session_id": session_id, "finished": True})

@app.post("/continue")
async def continue_generation(request: GenerationRequest):
    """继续基于现有会话进行生成"""
    if not request.session_id or request.session_id not in model_states:
        logger.warning(f"尝试继续不存在的会话: {request.session_id}")
        raise HTTPException(status_code=404, detail="会话ID不存在，请先创建会话")
    
    logger.info(f"继续生成: session={request.session_id}, max_length={request.max_length}, stream={request.stream}")
    return await generate_text(request)

@app.get("/sessions", response_model=SessionsResponse)
async def list_sessions():
    """列出所有活跃的会话ID"""
    logger.info(f"查询会话列表，当前会话数: {len(model_states)}")
    return {"sessions": list(model_states.keys())}

@app.delete("/session")
async def delete_session(request: SessionRequest):
    """删除特定会话"""
    if request.session_id in model_states:
        del model_states[request.session_id]
        logger.info(f"已删除会话: {request.session_id}")
        return {"message": f"会话 {request.session_id} 已删除"}
    else:
        logger.warning(f"尝试删除不存在的会话: {request.session_id}")
        raise HTTPException(status_code=404, detail="会话ID不存在")

@app.delete("/sessions")
async def clear_sessions(background_tasks: BackgroundTasks):
    """清除所有会话并释放内存"""
    logger.info(f"清理所有会话，当前会话数: {len(model_states)}")
    background_tasks.add_task(clear_memory)
    return {"message": "正在清理所有会话和内存"}

def clear_memory():
    """清理内存的辅助函数"""
    global model_states
    session_count = len(model_states)
    model_states.clear()
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"内存清理完成，释放了 {session_count} 个会话的状态")

if __name__ == "__main__":
    uvicorn.run(
        "run_rwkv_server:app", 
        host="0.0.0.0", 
        port=int(os.environ.get("RWKV_SERVER_PORT", 8001)),
        reload=False,
        log_level="info"
    )