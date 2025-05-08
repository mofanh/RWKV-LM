#!/usr/bin/env python3
"""
RWKV-7 API客户端示例
演示如何使用API进行文本生成，包括常规请求和流式请求
"""

import requests
import json
import time
import argparse
import sys
from typing import Dict, Any, Optional

def generate_text(
    prompt: str, 
    server_url: str = "http://localhost:8001", 
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.7,
    session_id: Optional[str] = None,
    stream: bool = False
) -> Dict[str, Any]:
    """
    调用RWKV-7 API生成文本
    
    参数:
        prompt: 提示文本
        server_url: 服务器URL
        max_length: 生成的最大长度
        temperature: 温度参数
        top_p: top-p参数
        session_id: 会话ID（可选）
        stream: 是否使用流式输出
        
    返回:
        常规模式: 包含生成文本的响应字典
        流式模式: 无返回值，直接打印文本
    """
    endpoint = f"{server_url}/generate"
    
    payload = {
        "prompt": prompt,
        "max_length": max_length,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream
    }
    
    if session_id:
        payload["session_id"] = session_id
        
    # 常规请求模式
    if not stream:
        response = requests.post(endpoint, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            raise Exception(f"请求失败，状态码: {response.status_code}, 原因: {response.text}")
    
    # 流式请求模式
    else:
        response = requests.post(endpoint, json=payload, stream=True)
        
        if response.status_code == 200:
            print("\n生成中: ", end="", flush=True)
            for chunk in response.iter_lines():
                if chunk:
                    data = json.loads(chunk.decode('utf-8'))
                    if not data.get("finished", False):
                        print(data["token"], end="", flush=True)
            print("\n生成完成!")
        else:
            raise Exception(f"请求失败，状态码: {response.status_code}, 原因: {response.text}")

def list_sessions(server_url: str = "http://localhost:8001") -> Dict[str, Any]:
    """获取所有活跃的会话ID"""
    response = requests.get(f"{server_url}/sessions")
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"获取会话列表失败，状态码: {response.status_code}, 原因: {response.text}")
        
def get_server_status(server_url: str = "http://localhost:8001") -> Dict[str, Any]:
    """获取服务器状态"""
    response = requests.get(f"{server_url}/status")
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"获取服务器状态失败，状态码: {response.status_code}, 原因: {response.text}")

def main():
    parser = argparse.ArgumentParser(description="RWKV-7 API客户端示例")
    parser.add_argument("--url", type=str, default="http://localhost:8001", help="服务器URL")
    parser.add_argument("--prompt", type=str, help="提示文本")
    parser.add_argument("--length", type=int, default=100, help="生成的最大长度")
    parser.add_argument("--temp", type=float, default=0.7, help="温度参数")
    parser.add_argument("--top_p", type=float, default=0.7, help="top-p参数")
    parser.add_argument("--session", type=str, help="会话ID")
    parser.add_argument("--stream", action="store_true", help="使用流式输出")
    parser.add_argument("--status", action="store_true", help="获取服务器状态")
    parser.add_argument("--sessions", action="store_true", help="列出所有会话")
    
    args = parser.parse_args()
    
    # 获取服务器状态
    if args.status:
        try:
            status = get_server_status(args.url)
            print("服务器状态:")
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return
        except Exception as e:
            print(f"错误: {str(e)}")
            return
    
    # 列出所有会话
    if args.sessions:
        try:
            sessions = list_sessions(args.url)
            print("当前活跃会话:")
            print(json.dumps(sessions, indent=2, ensure_ascii=False))
            return
        except Exception as e:
            print(f"错误: {str(e)}")
            return
    
    # 生成文本
    if args.prompt:
        try:
            if not args.stream:
                start_time = time.time()
                result = generate_text(
                    args.prompt, 
                    args.url, 
                    args.length, 
                    args.temp, 
                    args.top_p, 
                    args.session,
                    False
                )
                print("\n生成文本:")
                print(result["generated_text"])
                print(f"\n会话ID: {result['session_id']}")
                print(f"服务器执行时间: {result['execution_time']:.2f}s")
                print(f"总请求时间: {time.time() - start_time:.2f}s")
            else:
                generate_text(
                    args.prompt, 
                    args.url, 
                    args.length, 
                    args.temp, 
                    args.top_p, 
                    args.session,
                    True
                )
        except Exception as e:
            print(f"错误: {str(e)}")
            return
    else:
        parser.print_help()

if __name__ == "__main__":
    main()