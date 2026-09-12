const path = require('node:path');

function inside(directory, parent) {
  const relative = path.relative(path.resolve(parent), path.resolve(directory));
  return relative === '' || (!relative.startsWith('..' + path.sep) && relative !== '..' && !path.isAbsolute(relative));
}

class ProjectAccessCancelled extends Error {
  constructor() {
    super('已取消项目库访问授权；没有启动引擎或更换项目库。重新打开 App 可再次选择。');
    this.name = 'ProjectAccessCancelled';
  }
}

// An external configured library can be in macOS-protected Documents/Desktop.
// A Python child cannot reliably present that permission prompt before the App
// has a window. NSOpenPanel grants normal user-selected folder access first.
// Do not probe with readdir/glob beforehand: that was the blocking operation.
async function authorizeProjectDirectory({ data, userData, dialog, platform = process.platform, t = text => text }) {
  const expected = path.resolve(data);
  if (platform !== 'darwin' || inside(expected, userData)) return expected;
  let explanation = t('此 App 需要读取已配置的现有项目库。请选择下方同一文件夹，以授予 macOS 文件访问权限。项目和模型将保留原位置。');
  for (;;) {
    const result = await dialog.showOpenDialog({
      title: t('授权打开现有项目库'),
      message: `${explanation}\n\n${t('项目库：')}${expected}`,
      defaultPath: expected,
      buttonLabel: t('使用此项目库'),
      properties: ['openDirectory', 'dontAddToRecent'],
    });
    if (!result.canceled && result.filePaths?.length === 1 && path.resolve(result.filePaths[0]) === expected) return expected;
    const canceled = result.canceled || !result.filePaths?.length;
    const response = await dialog.showMessageBox({
      type: canceled ? 'info' : 'warning',
      title: t('尚未打开项目库'),
      message: canceled ? t('未选择项目库，本次尚未启动引擎。') : t('选择的文件夹与已配置项目库不同。'),
      detail: `${t('需要选择：')}${expected}\n${t('不会自动改用其他文件夹、复制数据或覆盖现有项目。')}`,
      buttons: [t('重新选择'), t('退出')], defaultId: 0, cancelId: 1, noLink: true,
    });
    if (response.response !== 0) throw new ProjectAccessCancelled();
    explanation = t('请在选择窗口中确认已配置的项目库文件夹；选择其他目录不会更换现有项目库。');
  }
}

module.exports = { authorizeProjectDirectory, ProjectAccessCancelled };
