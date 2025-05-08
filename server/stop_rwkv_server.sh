#!/bin/bash

# RWKV-7 API服务停止脚本

if [ -f rwkv_server.pid ]; then
    PID=$(cat rwkv_server.pid)
    
    if ps -p $PID > /dev/null; then
        echo "正在停止RWKV-7 API服务 (PID: $PID)..."
        kill $PID
        
        # 等待进程终止
        for i in {1..10}; do
            if ! ps -p $PID > /dev/null; then
                echo "RWKV-7 API服务已停止"
                rm rwkv_server.pid
                exit 0
            fi
            sleep 1
        done
        
        # 如果进程未能正常终止，使用强制终止
        echo "服务未能正常终止，尝试强制终止..."
        kill -9 $PID
        if ! ps -p $PID > /dev/null; then
            echo "RWKV-7 API服务已强制停止"
            rm rwkv_server.pid
            exit 0
        else
            echo "无法停止服务，请手动处理 PID: $PID"
            exit 1
        fi
    else
        echo "PID文件存在，但进程不存在，可能服务已经停止"
        rm rwkv_server.pid
        exit 0
    fi
else
    echo "找不到PID文件，RWKV-7 API服务可能未运行"
    exit 0
fi