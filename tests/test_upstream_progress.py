"""Exercise streamed CUDA-style progress without a CUDA process or model."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.upstream import run_logged


def test_tqdm_carriage_returns_and_stage_changes_do_not_keep_training_eta(tmp_path):
    script=tmp_path/'fake_trainer.py'
    script.write_text('''import sys
for line in ["Loading cameras 10/100", "Training progress: 10%|x| 10/100 [00:01<00:09, 10.00it/s]", "[ITER 10] Saving Gaussians", "[ITER 10] Evaluating test: L1 0.1", "Training progress: 100%|x| 100/100 [00:10<00:00, 10.00it/s]"]:
    sys.stdout.write(line + "\\r")
    sys.stdout.flush()
''')
    events=[]
    run_logged([sys.executable,str(script)],tmp_path,tmp_path/'train.log',lambda value,message:events.append((value,message)),lambda:False,'原始 3DGS CUDA 优化',100)
    assert len(events)==4
    assert 'Training progress:' in events[0][1]
    assert '保存' in events[1][1] and 'Training progress:' not in events[1][1]
    assert '验证' in events[2][1] and 'Training progress:' not in events[2][1]
    assert abs(events[3][0]-.92)<1e-9 and '训练迭代完成' in events[3][1]
