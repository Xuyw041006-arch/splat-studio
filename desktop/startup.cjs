const fs = require('node:fs');
const path = require('node:path');

// Optional local deployment settings. Relative dataPath values are resolved
// beside the application, so a test distribution can be moved as one folder.
// Ordinary distributions omit this file and retain the per-user data location.
function startupOptions({ env = process.env, isPackaged, resourcesPath, userData, platform = process.platform, cwd = process.cwd() }) {
  let config = {};
  let configBase = cwd;
  if (isPackaged) {
    const filename = path.join(resourcesPath, 'desktop-startup.json');
    configBase = platform === 'darwin' ? path.resolve(resourcesPath, '../../..') : path.dirname(resourcesPath);
    if (fs.existsSync(filename)) {
      if (fs.statSync(filename).size > 16384) throw new Error('桌面启动配置过大。');
      config = JSON.parse(fs.readFileSync(filename, 'utf8'));
      if (!config || Array.isArray(config) || typeof config !== 'object' || Object.keys(config).some(key => !['dataPath', 'initialResult'].includes(key))) {
        throw new Error('桌面启动配置只支持 dataPath 和 initialResult。');
      }
    }
  }
  const configuredData = env.SPLAT_DATA_DIR || config.dataPath;
  if (configuredData !== undefined && (typeof configuredData !== 'string' || !configuredData.trim() || configuredData.includes('\0'))) {
    throw new Error('桌面项目目录无效。');
  }
  const data = configuredData ? path.resolve(env.SPLAT_DATA_DIR ? cwd : configBase, configuredData) : path.join(userData, 'projects');
  const initialResult = env.SPLAT_INITIAL_RESULT || config.initialResult || '';
  if (typeof initialResult !== 'string' || (initialResult && !/^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(initialResult))) {
    throw new Error('桌面启动结果标识无效。');
  }
  return { data, initialResult };
}

module.exports = { startupOptions };
