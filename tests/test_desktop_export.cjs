const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { createSceneExporter, installSceneExport } = require('../desktop/scene-export.cjs');

async function fixture(t, options = {}) {
  const directory = await fs.mkdtemp(path.join(os.tmpdir(), 'splat-export-'));
  t.after(() => fs.rm(directory, { recursive: true, force: true }));
  const destination = path.join(directory, 'complete.splat.jsonl');
  const exporter = createSceneExporter({ choosePath: async () => destination, ...options });
  return { directory, destination, exporter };
}

test('streamed export commits only after finish and preserves exact Unicode chunks', async t => {
  const { exporter, destination, directory } = await fixture(t);
  await fs.writeFile(destination, 'original');
  const token = await exporter.begin(1, 'complete.splat.jsonl');
  await exporter.write(1, token, '{"header":"完整"}\n');
  await exporter.write(1, token, '{"sh":[0,1,-2,3]}\n');
  assert.equal(await fs.readFile(destination, 'utf8'), 'original');
  const result = await exporter.finish(1, token);
  const text = '{"header":"完整"}\n{"sh":[0,1,-2,3]}\n';
  assert.equal(await fs.readFile(destination, 'utf8'), text);
  assert.equal(result.bytes, Buffer.byteLength(text));
  assert.deepEqual(await fs.readdir(directory), ['complete.splat.jsonl']);
  await assert.rejects(exporter.write(1, token, 'late'), /会话/);
});

test('another window cannot write, finish, or abort the owner export', async t => {
  const { exporter } = await fixture(t);
  const token = await exporter.begin(4, 'scene.splat.jsonl');
  await assert.rejects(exporter.write(5, token, 'bad'), /不属于/);
  await assert.rejects(exporter.finish(5, token), /不属于/);
  await assert.rejects(exporter.abort(5, token), /不属于/);
  await exporter.abort(4, token);
});

test('capacity failure never commits a truncated file and cleans its temporary file', async t => {
  const { exporter, destination, directory } = await fixture(t, { maxChunkBytes: 5, maxTotalBytes: 8 });
  await fs.writeFile(destination, 'keep');
  const token = await exporter.begin(1, 'scene.splat.jsonl');
  await exporter.write(1, token, '12345');
  await assert.rejects(exporter.write(1, token, '6789'), /容量/);
  await assert.rejects(exporter.finish(1, token), /未完整/);
  assert.equal(await fs.readFile(destination, 'utf8'), 'keep');
  assert.deepEqual(await fs.readdir(directory), ['complete.splat.jsonl']);
});

test('cancel creates no files and closing a window removes an unfinished export', async t => {
  const canceled = createSceneExporter({ choosePath: async () => null });
  assert.equal(await canceled.begin(1, 'scene.splat.jsonl'), null);
  const { exporter, directory } = await fixture(t);
  const token = await exporter.begin(1, 'scene.splat.jsonl');
  await exporter.write(1, token, 'partial');
  await exporter.abortOwner(1);
  assert.deepEqual(await fs.readdir(directory), []);
});

test('IPC accepts only the owning local main frame before opening a save dialog', async () => {
  const handlers = new Map();
  let prompts = 0;
  const frame = { url: 'http://127.0.0.1:54321/?result=bonsai-sh3' };
  const sender = { id: 77, mainFrame: frame };
  installSceneExport({ ipcMain: { handle: (name, action) => handlers.set(name, action) },
    dialog: { showSaveDialog: async () => { prompts++; return { canceled: true }; } },
    win: { webContents: sender, on() {}, isDestroyed: () => false }, origin: 'http://127.0.0.1:54321' });
  const begin = handlers.get('scene-export:begin');
  assert.throws(() => begin({ sender: { id: 77 }, senderFrame: frame }, 'scene.splat.jsonl'), /不属于/);
  assert.throws(() => begin({ sender, senderFrame: { ...frame } }, 'scene.splat.jsonl'), /不属于/);
  frame.url = 'https://untrusted.invalid/';
  assert.throws(() => begin({ sender, senderFrame: frame }, 'scene.splat.jsonl'), /不属于/);
  assert.equal(prompts, 0);
  frame.url = 'http://127.0.0.1:54321/';
  assert.equal(await begin({ sender, senderFrame: frame }, 'scene.splat.jsonl'), null);
  assert.equal(prompts, 1);
});
