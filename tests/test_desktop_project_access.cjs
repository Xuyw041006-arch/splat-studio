const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { authorizeProjectDirectory, ProjectAccessCancelled } = require('../desktop/project-access.cjs');

function mock(picks, responses = []) {
  const calls = [];
  return { calls, dialog: {
    async showOpenDialog(options) { calls.push(['open', options]); return picks.shift(); },
    async showMessageBox(options) { calls.push(['message', options]); return { response: responses.shift() }; },
  } };
}

test('macOS external library requires exact native directory selection before use', async () => {
  const data = '/Users/test/Documents/Scenes'; const state = mock([{ canceled: false, filePaths: [data] }]);
  assert.equal(await authorizeProjectDirectory({ data, userData: '/Users/test/Library/Application Support/Splat', platform: 'darwin', dialog: state.dialog }), data);
  assert.equal(state.calls.length, 1);
  assert.equal(state.calls[0][1].defaultPath, data);
  assert.deepEqual(state.calls[0][1].properties, ['openDirectory', 'dontAddToRecent']);
  assert.match(state.calls[0][1].message, /保留原位置/);
});

test('Windows and ordinary App userData do not add macOS permission flows', async () => {
  const state = mock([]);
  for (const values of [
    { platform: 'win32', data: '/external/scenes', userData: '/app/data' },
    { platform: 'darwin', data: '/app/data/projects', userData: '/app/data' },
    { platform: 'darwin', data: '/app/data', userData: '/app/data' },
  ]) assert.equal(await authorizeProjectDirectory({ ...values, dialog: state.dialog }), values.data);
  assert.equal(state.calls.length, 0);
});

test('similar prefix is not treated as the private userData directory', async () => {
  const data = '/app/data-external/projects';const state = mock([{ canceled: false, filePaths: [data] }]);
  await authorizeProjectDirectory({ data, userData: '/app/data', platform: 'darwin', dialog: state.dialog });
  assert.equal(state.calls.length, 1);
});

test('wrong folder is rejected; retry keeps the configured library unchanged', async () => {
  const data = '/configured/library';const state = mock([
    { canceled: false, filePaths: ['/different/library'] }, { canceled: false, filePaths: [data] },
  ], [0]);
  assert.equal(await authorizeProjectDirectory({ data, userData: '/app/data', platform: 'darwin', dialog: state.dialog }), data);
  assert.deepEqual(state.calls.filter(c => c[0] === 'open').map(c => c[1].defaultPath), [data, data]);
  assert.match(state.calls[1][1].message, /不同/);
});

test('cancel with exit explicitly stops instead of switching or starting a library', async () => {
  const state = mock([{ canceled: true, filePaths: [] }], [1]);
  await assert.rejects(authorizeProjectDirectory({ data: '/configured/library', userData: '/app/data', platform: 'darwin', dialog: state.dialog }), ProjectAccessCancelled);
  assert.equal(state.calls.length, 2);
});

test('cancel supports a deliberate retry and multiple directory values never pass', async () => {
  const data = '/configured/library';const state = mock([
    { canceled: true, filePaths: [] }, { canceled: false, filePaths: [data, '/other'] },
    { canceled: false, filePaths: [data] },
  ], [0, 0]);
  assert.equal(await authorizeProjectDirectory({ data, userData: '/app/data', platform: 'darwin', dialog: state.dialog }), data);
  assert.equal(state.calls.filter(c => c[0] === 'open').length, 3);
});

test('desktop startup awaits folder authorization before creating data or spawning backend', () => {
  const source = fs.readFileSync(path.join(__dirname, '../desktop/main.cjs'), 'utf8');
  const authorize = source.indexOf('await authorizeProjectDirectory(');
  assert.ok(authorize > 0 && authorize < source.indexOf('fs.mkdirSync(data,'));
  assert.ok(authorize < source.indexOf('backend=spawn('));
  assert.match(source, /error instanceof ProjectAccessCancelled/);
});
