const nodePath = require('node:path');

// Keep external CUDA environments separate from the app's embedded CPU Python.
function backendRuntime({ platform, isPackaged, resourcesPath, root, env, exists }) {
  const path = platform === 'win32' ? nodePath.win32 : nodePath.posix;
  if (isPackaged && env.SPLAT_PYTHON) {
    if (!path.isAbsolute(env.SPLAT_PYTHON) || !exists(env.SPLAT_PYTHON)) {
      throw new Error('SPLAT_PYTHON 需要指向已配置训练环境的 Python 完整路径。');
    }
    const runtime = path.join(resourcesPath, 'backend', 'splat-backend', '_internal', 'cloud-runtime');
    return { command: env.SPLAT_PYTHON, args: ['-m', 'backend.app'],
      env: { PYTHONPATH: runtime + (env.PYTHONPATH ? path.delimiter + env.PYTHONPATH : '') } };
  }
  if (isPackaged) {
    const embedded = path.join(resourcesPath, 'backend', 'python', 'python.exe');
    if (platform === 'win32' && exists(embedded)) {
      return { command: embedded, args: ['-I', '-B', '-X', 'utf8', path.join(resourcesPath, 'backend', 'run_portable.py')], env: {} };
    }
    return { command: path.join(resourcesPath, 'backend', 'splat-backend', platform === 'win32' ? 'splat-backend.exe' : 'splat-backend'), args: [], env: {} };
  }
  const local = path.join(root, '.venv', platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  return { command: env.SPLAT_PYTHON || (exists(local) ? local : (platform === 'win32' ? 'python' : 'python3')),
    args: ['-m', 'backend.app'], env: {} };
}
module.exports = { backendRuntime };
