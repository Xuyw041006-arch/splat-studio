const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { startupOptions } = require('../desktop/startup.cjs');

test('normal desktop keeps user projects and explicit environment overrides are portable', () => {
  const options = { env: {}, isPackaged: false, userData: '/tmp/splat-user', cwd: '/tmp/splat-work' };
  assert.deepEqual(startupOptions(options), { data: '/tmp/splat-user/projects', initialResult: '' });
  assert.deepEqual(startupOptions({ ...options, env: { SPLAT_DATA_DIR: 'results', SPLAT_INITIAL_RESULT: 'bonsai-sh3' } }), { data: '/tmp/splat-work/results', initialResult: 'bonsai-sh3' });
});

test('packaged data configuration resolves relative to the Mac application directory', () => {
  const base = fs.mkdtempSync(path.join(os.tmpdir(), 'splat-startup-'));
  try {
    const resourcesPath = path.join(base, 'Splat Studio Test.app/Contents/Resources');
    fs.mkdirSync(resourcesPath, { recursive: true });
    const filename = path.join(resourcesPath, 'desktop-startup.json');
    const options = { env: {}, isPackaged: true, resourcesPath, platform: 'darwin', userData: path.join(base, 'user') };
    fs.writeFileSync(filename, JSON.stringify({ dataPath: 'results/projects', initialResult: 'bonsai-sh3' }));
    assert.deepEqual(startupOptions(options), { data: path.join(base, 'results/projects'), initialResult: 'bonsai-sh3' });
    assert.equal(startupOptions({ ...options, env: { SPLAT_DATA_DIR: '/tmp/explicit', SPLAT_INITIAL_RESULT: 'teatime-r2' } }).data, '/tmp/explicit');
    fs.writeFileSync(filename, JSON.stringify({ dataPath: 'results', command: 'ignored' }));
    assert.throws(() => startupOptions(options), /只支持/);
    fs.writeFileSync(filename, JSON.stringify({ initialResult: '../invalid' }));
    assert.throws(() => startupOptions(options), /结果标识/);
    fs.writeFileSync(filename, JSON.stringify({ dataPath: [] }));
    assert.throws(() => startupOptions(options), /目录无效/);
  } finally { fs.rmSync(base, { recursive: true, force: true }); }
});
