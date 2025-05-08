chmod +x start_rwkv_server.sh
chmod +x stop_rwkv_server.sh
chmod +x client_example.py



# 使用默认端口(8001)和默认GPU(0)
./start_rwkv_server.sh

# 或指定端口和GPU
./start_rwkv_server.sh 8000 1



# 获取服务器状态
./client_example.py --status

# 列出所有会话
./client_example.py --sessions

# 常规生成
./client_example.py --prompt "System: 你是一个专业的中医医生，正在进行一次完整的中医诊疗对话。\n\nUser: 我是患者，请问咳嗽有痰该怎么调理？" --length 150

# 流式生成
./client_example.py --prompt "System: 你是一个专业的中医医生，正在进行一次完整的中医诊疗对话。\n\nUser: 我是患者，请问咳嗽有痰该怎么调理？" --length 150 --stream

# 使用特定会话ID继续生成
./client_example.py --prompt "User: 这些方法需要注意什么？" --session "session_1620000000" --stream



./stop_rwkv_server.sh