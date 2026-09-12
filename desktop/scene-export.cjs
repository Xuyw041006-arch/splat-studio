const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');

// Renderer receives opaque handles, never filesystem paths or a generic IPC API.
function createSceneExporter({ choosePath, maxChunkBytes = 64 * 1024 ** 2, maxTotalBytes = 4 * 1024 ** 3 }) {
  const records = new Map();
  const choosing = new Set();
  function owned(owner, token) {
    const row = records.get(token);
    if (!row || row.owner !== owner || row.closing) throw new Error('导出会话已结束或不属于此窗口。');
    return row;
  }
  function enqueue(row, action) {
    const result = row.queue.then(action);
    row.queue = result.catch(() => { row.failed = true; });
    return result;
  }
  async function discard(token, row) {
    await row.queue;
    await row.file.close().catch(() => {});
    await fs.unlink(row.temporary).catch(() => {});
    records.delete(token);
  }
  return {
    async begin(owner, suggestedName) {
      if (choosing.has(owner) || [...records.values()].some(row => row.owner === owner)) throw new Error('此窗口已有导出任务。');
      const name = typeof suggestedName === 'string' && suggestedName.length <= 160 && !/[\\/\0]/.test(suggestedName)
        ? suggestedName : 'scene.splat.jsonl';
      choosing.add(owner);
      try {
        const destination = await choosePath(name);
        if (!destination) return null;
        if (!path.isAbsolute(destination)) throw new Error('导出路径必须来自原生保存对话框。');
        const token = crypto.randomUUID();
        const temporary = path.join(path.dirname(destination), `.${path.basename(destination)}.${token}.partial`);
        const file = await fs.open(temporary, 'wx', 0o600);
        records.set(token, { owner, destination, temporary, file, bytes: 0, queue: Promise.resolve(), failed: false, closing: false });
        return token;
      } finally { choosing.delete(owner); }
    },
    async write(owner, token, text) {
      const row = owned(owner, token);
      if (typeof text !== 'string' || Buffer.byteLength(text) > maxChunkBytes || row.bytes + Buffer.byteLength(text) > maxTotalBytes) {
        row.failed = true;
        throw new Error('导出超过单块 64 MiB 或总文件 4 GiB 容量限制；不会截断或覆盖目标文件。');
      }
      row.bytes += Buffer.byteLength(text);
      return enqueue(row, async () => {
        if (row.failed) throw new Error('导出已失败，请重新导出。');
        await row.file.writeFile(text, 'utf8');
      });
    },
    async finish(owner, token) {
      const row = owned(owner, token); row.closing = true;
      try {
        await row.queue;
        if (row.failed || row.bytes === 0) throw new Error('导出未完整写入，目标文件保持原样。');
        await row.file.sync(); await row.file.close();
        await fs.rename(row.temporary, row.destination);
        records.delete(token);
        return { bytes: row.bytes };
      } catch (error) { await discard(token, row); throw error; }
    },
    async abort(owner, token) {
      const row = owned(owner, token); row.closing = true;
      await discard(token, row);
    },
    async abortOwner(owner) {
      await Promise.all([...records].filter(([, row]) => row.owner === owner).map(async ([token, row]) => {
        row.closing = true; await discard(token, row);
      }));
    },
  };
}

function installSceneExport({ ipcMain, dialog, win, origin, t = text => text }) {
  const owner = win.webContents.id;
  const exporter = createSceneExporter({ choosePath: async defaultPath => {
    const result = await dialog.showSaveDialog(win, { title: t('导出完整高斯场景'), defaultPath,
      filters: [{ name: t('Splat Studio 场景'), extensions: ['jsonl'] }],
      properties: ['showOverwriteConfirmation', 'createDirectory'] });
    return result.canceled || win.isDestroyed() ? null : result.filePath;
  } });
  function authorize(event) {
    if (event.sender !== win.webContents || event.senderFrame !== win.webContents.mainFrame || new URL(event.senderFrame.url).origin !== origin) {
      throw new Error('导出请求不属于当前本地工作台。');
    }
  }
  const methods = { begin: 'begin', write: 'write', finish: 'finish', abort: 'abort' };
  for (const [channel, method] of Object.entries(methods)) ipcMain.handle(`scene-export:${channel}`, (event, ...args) => {
    authorize(event); return exporter[method](owner, ...args);
  });
  win.on('closed', () => { void exporter.abortOwner(owner); });
}

module.exports = { createSceneExporter, installSceneExport };
