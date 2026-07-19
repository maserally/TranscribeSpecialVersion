# 云算力部署（AutoDL / Linux GPU）

## 推荐：本地工作室连接云 GPU 运算单元

Git 拉取本身不会安装任何依赖。本项目现在由本地工作室主动连接云服务器并完成云端运算依赖安装，因此云服务器不需要手动克隆仓库：

1. 在 Windows 本地更新项目并安装 `requirements.txt`。
2. 启动本地字幕工作室，找到“云 GPU 运算单元”。
3. 填写云实例的 SSH 地址、端口、用户名，以及密码或私钥路径。
4. 点击“测试连接”，确认能读取 NVIDIA GPU。
5. 点击“连接并安装运算环境”。程序会自动检查并安装 Python venv、FFmpeg、NumPy 和 Whisper；已安装过的节点会使用就绪标记跳过重复安装。
6. 勾选“使用云 GPU”并正常创建任务。

## 无卡模式预上传

需要批量处理时，可以先把 AutoDL 切换为无卡模式，再在本地工作室中：

1. 更新并保存无卡实例当前显示的 SSH 地址和端口，点击“测试连接”。
2. 选择单个视频，或扫描文件夹并勾选需要处理的视频。
3. 点击“无卡模式预上传音轨”。程序只在本地提取 `16 kHz` 单声道 FLAC，不安装模型，也不检查 CUDA。
4. 等所有任务显示“已预上传”。上传先写入云端 `.uploading` 临时文件；云端重新计算文件大小和 SHA-256，两项均与本地一致后才原子发布为正式音轨。
5. 关闭无卡实例，按正常 GPU 模式开机。如果 SSH 地址或端口变化，先更新配置并再次测试连接。
6. 点击“开始全部已预上传”。每个任务在进入识别前会再次计算云端实际文件的大小和 SHA-256；不一致时自动重新上传，连续三次失败则拒绝识别。

上传期间如果 SFTP 连接中断，程序会重新连接，先校验云端 `.uploading` 临时文件与本地音轨相同长度前缀的 SHA-256，再从已确认字节处继续上传。失败任务可点击“断点续传”，也可使用任务栏的“重试失败预上传”批量恢复；新版失败任务会保留本地 FLAC，不需要重新提取音轨或从 0% 开始。

预上传成功后，本地任务目录仍保留提取出的 FLAC，原视频也不会删除。删除“已预上传”任务时，程序会先清理对应云端目录；如果云节点不可连接，会拒绝删除本地任务，以免误以为云端音轨已经清除。

任务的数据流如下：

```text
本地视频 → 本地提取 16 kHz 单声道 FLAC → 本地轻量 VAD
        → 上传音轨和 VAD 分段 → 云 GPU 执行声音分类、主识别、复核和弱对白召回
        → 下载识别 JSON → 本地翻译、自动语境校正、生成 SRT/视频
```

原视频、API Key、翻译配置和本地任务历史不会上传到云节点。云端任务临时目录在识别结束或任务取消后自动清理，Whisper 模型缓存和虚拟环境会保留，便于下次复用。

建议使用 AutoDL 已预装 CUDA 与 PyTorch 的镜像，并使用实例提供的 `root` SSH 账号。通用 Linux 非 root 账号如果缺少 FFmpeg 或 `python3-venv`，需要具有免密码 sudo 权限，否则自动安装系统包会失败并给出错误。

以下“完整云端网页模式”仍然保留，适合希望整个网页服务都运行在云服务器的情况。

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
