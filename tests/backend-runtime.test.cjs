const test = require('node:test');
const assert = require('node:assert/strict');
const { backendRuntime } = require('../desktop/backend-runtime.cjs');
const base = { platform: 'win32', isPackaged: true, resourcesPath: 'C:\\用户 文件\\Splat Studio\\resources', root: 'C:\\source', env: {}, exists: () => true };
test('portable Windows launch preserves spaces and Unicode as arguments', () => {
  const r = backendRuntime(base);
  assert.equal(r.command, base.resourcesPath+'\\backend\\python\\python.exe');
  assert.deepEqual(r.args, ['-I','-B','-X','utf8',base.resourcesPath+'\\backend\\run_portable.py']);
});
test('explicit CUDA override takes precedence over CPU Python', () => {
  const r = backendRuntime({...base,env:{SPLAT_PYTHON:'D:\\GPU env\\Scripts\\python.exe',PYTHONPATH:'D:\\shared'}});
  assert.equal(r.command,'D:\\GPU env\\Scripts\\python.exe');
  assert.deepEqual(r.args,['-m','backend.app']);
  assert.ok(r.env.PYTHONPATH.endsWith(';D:\\shared'));
});
test('invalid explicit CUDA Python fails instead of silently using CPU', () => {
  assert.throws(()=>backendRuntime({...base,env:{SPLAT_PYTHON:'python.exe'}}));
  assert.throws(()=>backendRuntime({...base,exists:()=>false,env:{SPLAT_PYTHON:'C:\\gone\\python.exe'}}));
});
test('native frozen packages remain supported on both operating systems', () => {
  assert.equal(backendRuntime({...base,exists:()=>false}).command,base.resourcesPath+'\\backend\\splat-backend\\splat-backend.exe');
  assert.equal(backendRuntime({...base,platform:'darwin',resourcesPath:'/Applications/Splat Studio.app/Contents/Resources'}).command,
    '/Applications/Splat Studio.app/Contents/Resources/backend/splat-backend/splat-backend');
});
