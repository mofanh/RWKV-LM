#!/bin/bash

# RWKV-7 API服务启动脚本
# 使用方法: ./start_rwkv_server.sh [port] [gpu_id]

# 设置默认值
PORT="${1:-8001}"
GPU_ID="${2:-0}"
LOG_FILE="rwkv_server_$(date +%Y%m%d_%H%M%S).log"

# 确保目录存在
mkdir -p logs

echo "正在启动RWKV-7 API服务..."
echo "端口: $PORT"
echo "使用GPU: $GPU_ID"
echo "日志文件: logs/$LOG_FILE"

# 设置环境变量
export RWKV_SERVER_PORT=$PORT
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 启动服务器
nohup python run_rwkv_server.py > logs/$LOG_FILE 2>&1 &

# 获取进程ID
PID=$!
echo "服务进程ID: $PID"
echo $PID > rwkv_server.pid

# 等待服务启动
sleep 5

# 检查服务是否正常运行
if ps -p $PID > /dev/null; then
    echo "RWKV-7 API服务已成功启动！"
    echo "可以通过 http://localhost:$PORT/docs 访问API文档"
else
    echo "服务启动失败，请检查日志: logs/$LOG_FILE"
    exit 1
fi