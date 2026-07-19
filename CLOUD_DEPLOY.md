# 云算力部署（AutoDL / Linux GPU）

## 推荐方式

租用带 NVIDIA GPU 的 Linux 实例。16 GB 显存可运行 `medium + large-v3`，24 GB 显存更宽裕。程序会自动复核背景音乐中的弱对白，因此会增加一些运行时间，但比本机 CPU 快得多。

不要直接把未加认证的网页端口暴露到公网。推荐让服务只监听 `127.0.0.1:6006`，再通过 SSH 隧道访问。

## 首次安装

```bash
git clone https://github.com/maserally/TranscribeSpecialVersion.git
cd TranscribeSpecialVersion
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get update && sudo apt-get install -y ffmpeg
```

如果镜像已经提供 PyTorch/CUDA，优先保留镜像匹配的 PyTorch，再安装其余依赖，避免把 CUDA 版覆盖成 CPU 版。

## 启动

```bash
source .venv/bin/activate
export SUBTITLE_STUDIO_DATA_DIR=/root/autodl-tmp/subtitle-studio-data
export SUBTITLE_TRANSLATOR_API_KEY='你的翻译接口密钥'
export SUBTITLE_REVIEWER_API_KEY='你的校正接口密钥'
bash start_cloud.sh
```

如果识别也调用云接口，再设置 `SUBTITLE_ASR_API_KEY`。Linux 云模式不会把 API Key 写入配置文件；网页只保存提供方、Base URL 和模型 ID。

本机建立隧道：

```bash
ssh -L 6006:127.0.0.1:6006 用户名@云主机地址
```

然后在本机浏览器打开 `http://127.0.0.1:6006`。AutoDL 的 SSH 地址和端口以实例页面为准。

## 素材与产物

- 云模式只显示上传入口，产物按钮自动变为“下载”。
- 建议本机先提取音频再上传；识别完成后下载 SRT，可节省大视频上传和云盘空间。
- 若上传的是音频，保持软字幕视频和硬字幕视频关闭。
- `SUBTITLE_STUDIO_DATA_DIR` 必须指向实例的持久化数据盘，否则释放实例后任务记录和模型缓存可能丢失。
- 程序会自动检查普通长空白复核覆盖不到的短弱对白，并要求两个 Whisper 模型达到一致性后才加入字幕，不需要手动开启补漏功能。

若确实需要监听 `0.0.0.0`，请同时设置强用户名和密码：

```bash
export SUBTITLE_STUDIO_HOST=0.0.0.0
export SUBTITLE_STUDIO_USERNAME='你的用户名'
export SUBTITLE_STUDIO_PASSWORD='足够长的随机密码'
```

即使启用基础认证，仍建议配合云平台防火墙、反向代理 HTTPS 或 SSH 隧道使用。
