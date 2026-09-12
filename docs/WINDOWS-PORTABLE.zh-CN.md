# Windows x64 便携发行

用户发行包为 Splat-Studio-0.4.2-Windows-x64.zip，包含实际 Windows Electron 和 CPython 嵌入式环境。整个文件夹必须一起分发。

在 Mac 上组装并校验，尚未执行 Windows 真机 / GPU 测试；不能把本机回归通过当成 Windows 功能认证。共享代码回归：712 项 Python + 17 个子测试、112 项 Node。

普通使用：安装随包的微软 VC++ x64 运行库，运行 Self-Test.cmd，再启动 Splat Studio.exe。内置 CPU 环境用于客户端和任务交换；CUDA 环境独立通过 SPLAT_PYTHON 配置。所有环境具体流程在发行目录的「完整使用说明.html」第 10 章。

维护构建：scripts/build_portable_windows.py 使用官方 Electron 38.8.6 win32-x64 ZIP、SHASUMS256.txt、Python 3.13.15 embed-amd64 ZIP、微软 VC_redist.x64.exe、Windows cp313/abi3 或通用 wheels。组件及 wheel SHA 在 WINDOWS-RUNTIME-LOCK.json。

以 pip download 的 --platform win_amd64 --python-version 3.13 --implementation cp --abi cp313 --only-binary=:all: 下载依赖；torch/torchvision 使用 PyTorch 官方 CPU 索引。selection 是锁文件中 wheel 文件名组成的 JSON 数组。frontend 输入为 pnpm run build 的 dist，geometry 输入包含已验证 dust3r 源码，不带模型权重。

python scripts/build_portable_windows.py --downloads 下载目录 --wheels wheel目录 --selection 文件名数组.json --frontend dist --geometry build/geometry --output 不存在的目录

构建脚本校验 Python/Electron SHA、wheel RECORD、目标平台依赖和 PE 架构。保留组件许可证。压缩前加入 START-HERE.txt 后，应更新 BUILD-MANIFEST.json 文件哈希。源码 CI 另有 Windows runner 的 PyInstaller/NSIS 路线，本次未执行该 CI，未生成 NSIS 安装器。

隔离 Python 用 -X utf8 避免 Windows 系统代码页影响。Windows 照片预检用 NumPy 文件读取 + OpenCV 解码，支持中文路径。环境脚本仅对自己的 Git 子进程关闭 CRLF 转换，保证固定训练源码校验一致。

应用右上角的 中文 / EN 可即时切换语言，原生保存对话框和下次启动也会使用所选语言。
