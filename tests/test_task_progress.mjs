import assert from 'node:assert/strict';
import test from 'node:test';
import {estimateBasisText, trainingEtaText} from '../frontend/task_progress.js';

const duration = seconds => `${seconds} 秒`;
const job = (progress = {}, status = 'running') => ({status, training_progress: {
  phase: 'training', iteration: 300, total: 1000, updated_at: 102,
  eta_seconds: 7, eta_basis: 'tqdm_observed_rate', scope: 'training_loop_only', ...progress,
}});

test('training ETA names its scope, source and excluded phases', () => {
  const text = trainingEtaText(job(), duration, 103000);
  assert.match(text, /训练循环剩余约 7 秒/);
  assert.match(text, /近期日志速率/);
  assert.match(text, /不含验证、保存和融合/);
  assert.match(trainingEtaText(job({eta_basis: 'observed_iteration_intervals'}), duration, 103000), /近期实测迭代/);
});

test('repeated polling cannot keep the last observed ETA alive beyond 45 seconds', () => {
  const unchanged = job();
  assert.match(trainingEtaText(unchanged, duration, 147000), /7 秒/);
  const expired = trainingEtaText(unchanged, duration, 147001);
  assert.match(expired, /ETA 已暂停/);
  assert.doesNotMatch(expired, /7 秒/);
});

test('stale, invalid and future timestamps cannot present numeric ETA', () => {
  for (const progress of [{phase: 'stale'}, {updated_at: undefined}, {updated_at: NaN}, {updated_at: 104}]) {
    const text = trainingEtaText(job(progress), duration, 103000);
    assert.match(text, /ETA 已暂停/);
    assert.doesNotMatch(text, /剩余约/);
  }
});

test('nonrunning jobs, completed loops and other phases clear ETA', () => {
  for (const status of ['queued', 'completed', 'failed', 'cancelled']) {
    assert.equal(trainingEtaText(job({}, status), duration, 103000), '');
  }
  for (const phase of ['saving', 'validating', 'initializing']) {
    assert.equal(trainingEtaText(job({phase}), duration, 103000), '');
  }
  assert.equal(trainingEtaText(job({iteration: 1000}), duration, 103000), '');
  assert.equal(trainingEtaText({status: 'running'}, duration, 103000), '');
});

test('collecting or nonfinite rates display no numeric remainder', () => {
  for (const eta_seconds of [null, undefined, NaN, Infinity]) {
    const text = trainingEtaText(job({eta_seconds}), duration, 103000);
    assert.match(text, /暂无稳定 ETA/);
    assert.doesNotMatch(text, /剩余约/);
  }
});

test('planning explanation distinguishes coarse planning from observed task time', () => {
  const text = estimateBasisText({estimate_basis: '本机小样本粗估，不含首次下载。'});
  assert.match(text, /非本任务实测/);
  assert.match(text, /不含首次下载/);
  assert.match(estimateBasisText({}), /缺少针对本场景的标定/);
});
