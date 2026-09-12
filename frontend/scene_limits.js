// A capacity guard, never a sampling budget. Both full benchmark models fit.
export const MAX_SCENE_GAUSSIANS = 2_000_000;

export function assertSceneCapacity(count, maximum = MAX_SCENE_GAUSSIANS) {
  if (!Number.isSafeInteger(maximum) || maximum < 1 || maximum > MAX_SCENE_GAUSSIANS) {
    throw new Error('完整场景读取上限无效。');
  }
  if (!Number.isSafeInteger(count) || count < 0) throw new Error('场景高斯总数无效。');
  if (count > maximum) {
    throw new Error(`场景总数无效：包含 ${count.toLocaleString()} 个高斯，超过当前完整显示上限 ${maximum.toLocaleString()}。没有抽样；请使用容量更大的渲染器或显式导出预览。`);
  }
}
