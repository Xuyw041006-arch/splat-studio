"""Validate evaluation science: split, no test RGB init, metrics, occlusion."""
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.run_benchmark import initialize_from_train, mask_metrics, render_scene, rgb_metrics, split_views
from benchmarks.evaluate_cuda_semantics import camera_for_record, read_training_masks
from benchmarks.evaluate_cuda_full_resolution import build_plan as full_resolution_plan, disjoint_output
from benchmarks.analyze_render_scale import compare_view
from benchmarks.collect_colab_delivery import choose_files, build_manifest, validate_completed, verify_archive


def test_split_excludes_labeled_evaluation_from_training(tmp_path):
    records = [{"name": f"{i:03}.png", "file": str(tmp_path / f"{i:03}.png")} for i in range(40)]
    train, test = split_views(records, 16, 8, annotated_names=["003.png", "009.png"])
    assert len(train) == 16
    assert {r["name"] for r in train}.isdisjoint({r["name"] for r in test})
    assert {"003.png", "009.png"} <= {r["name"] for r in test}


def test_no_test_color_initializes_manifest_points(tmp_path):
    image = tmp_path / "train.png"
    Image.new("RGB", (16, 16), (255, 0, 0)).save(image)
    record = {"file": str(image), "width": 16, "height": 16, "intrinsics": [[8,0,8],[0,8,8],[0,0,1]], "world_to_camera": np.eye(4).tolist()}
    raw = {"points": np.array([[0,0,2], [100,0,2]], float), "colors": np.array([[0,0,1], [0,0,1]], float)}
    points, colors, _, _ = initialize_from_train([record], raw, "manifest")
    assert len(points) == 1
    np.testing.assert_allclose(colors, [[1,0,0]])


def test_metrics_retain_failure_and_empty_roi():
    black = np.zeros((24,24,3), float)
    white = np.ones_like(black)
    assert rgb_metrics(black, white)["psnr_db"] == pytest.approx(0)
    assert rgb_metrics(white, white)["ssim"] == pytest.approx(1)
    assert rgb_metrics(white, white)["psnr_db"] == pytest.approx(120)
    assert rgb_metrics(white, white, np.zeros((24,24), bool))["psnr_db"] is None
    assert mask_metrics(np.zeros((3,3), bool), np.ones((3,3), bool))["iou"] == 0
    assert mask_metrics(np.zeros((3,3), bool), np.zeros((3,3), bool))["iou"] is None


def test_boundary_metric_and_truncated_objects():
    mask = np.zeros((32,32), bool)
    mask[4:28, 4:28] = True
    shifted = np.roll(mask, 2, axis=1)
    result = mask_metrics(shifted, mask)
    assert result["boundary_iou"] < result["iou"]
    assert not result["gt_touches_image_border"]
    mask[:, 0] = True
    result = mask_metrics(mask, mask)
    assert result["gt_touches_image_border"]
    assert result["boundary_iou"] == 1


def test_semantic_render_respects_unlabeled_front_occluder(tmp_path):
    image = tmp_path / "view.png"
    Image.new("RGB", (16,16), (0,0,0)).save(image)
    record = {"file": str(image), "width":16, "height":16, "intrinsics":[[8,0,8],[0,8,8],[0,0,1]], "world_to_camera":np.eye(4).tolist()}
    gaussian = lambda z: {"position":[0,0,z], "scale":[.4,.4,.4], "rotation":[1,0,0,0], "color":[.5,.5,.5], "opacity":.99}
    front, back = gaussian(2), gaussian(4)
    back["semantic_ids"] = ["target"]
    scene = {"gaussians":[front,back], "objects":[{"id":"target","label":"cup"}]}
    pred, _ = render_scene(scene, record, 16, "cpu", {"cup"})
    assert pred[8,8,0] < .02
    scene["gaussians"] = [back]
    pred, _ = render_scene(scene, record, 16, "cpu", {"cup"})
    assert pred[8,8,0] > .98


def test_cuda_camera_projection_preserves_off_center_intrinsics(tmp_path):
    import torch
    image = tmp_path / "view.png"
    Image.new("RGB", (32,24)).save(image)
    record = {"file":str(image),"name":"view.png","width":32,"height":24,
              "intrinsics":[[20,0,12],[0,22,11],[0,0,1]],"world_to_camera":np.eye(4).tolist()}
    camera, _ = camera_for_record(record,32,torch,device="cpu")
    xyzw = torch.tensor([.3,.2,2.,1.]) @ camera.full_proj_transform
    pixel = ((xyzw[:2]/xyzw[3]+1)/2 * torch.tensor([32,24])).numpy()
    np.testing.assert_allclose(pixel,[15,13.2],atol=1e-5)


def test_cuda_masks_cannot_include_a_test_view(tmp_path):
    import json
    mask = tmp_path / "mask.png"
    Image.new("L",(8,8),255).save(mask)
    config = tmp_path / "masks.json"
    config.write_text(json.dumps({"masks":[{"image_path":"test.jpg","mask_path":"mask.png","label":"cup"}]}))
    with pytest.raises(ValueError,match="actual training split"):
        read_training_masks(config,{"test.jpg":{"file":"test.jpg"}},{"train.jpg"})


def test_full_resolution_output_cannot_overwrite_input(tmp_path):
    original=tmp_path/"original"
    with pytest.raises(ValueError,match="separate"):
        disjoint_output(original,[original])
    with pytest.raises(ValueError,match="separate"):
        disjoint_output(original/"nested",[original])
    disjoint_output(tmp_path/"new",[original])


def test_full_resolution_plan_uses_seven_existing_checkpoints(tmp_path):
    import json
    baseline,priority=tmp_path/"baseline",tmp_path/"priority"
    baseline.mkdir()
    worker=baseline/"_upstream_training_worker.py"
    worker.write_text("# frozen worker fixture\n")
    for scene,size in [("bonsai",(780,520)),("teatime",(988,730))]:
        data=tmp_path/"data"/scene
        (data/"images").mkdir(parents=True)
        Image.new("RGB",size).save(data/"images/test.png")
        modes=[("fast",7000,baseline/scene/"fast"),("balanced",15000,baseline/scene/"balanced"),("fine",22000,baseline/scene/"fine")]
        if scene=="teatime":
            modes.append(("balanced",15000,priority/"balanced"))
        for mode,iterations,model in modes:
            cloud=model/"point_cloud"/f"iteration_{iterations}"/"point_cloud.ply"
            cloud.parent.mkdir(parents=True)
            cloud.write_text("fixture only; dry plan never reads or renders PLY contents")
            split={"train":["train.png"],"test":["test.png"]}
            job={"scene":scene,"mode":mode,"iterations":iterations,"split":split,"scene_dir":str(data),"images":"images","upstream":"unused","model_path":str(model)}
            result={"split":split,"timing":{"test_image_names":["test.png"]},"evaluation":{}}
            (model/"job.json").write_text(json.dumps(job))
            (model/"result.json").write_text(json.dumps(result))
    planned=full_resolution_plan(baseline,priority,tmp_path/"full-resolution",worker)
    assert len(planned)==7
    assert all(item["render_job"]["action"]=="render" for item in planned)
    assert [item["render_job"]["eval_width"] for item in planned]==[780]*3+[988]*4
    assert planned[-1]["mode"]=="balanced-priority"
    assert all(item["source_checkpoint_sha256"] for item in planned)


def test_render_scale_comparison_requires_identical_resized_gt(tmp_path):
    native_gt=tmp_path/"native-gt.png"
    original_gt=tmp_path/"original-gt.png"
    original_render=tmp_path/"original-render.png"
    Image.new("RGB",(512,384),(128,128,128)).save(native_gt)
    Image.open(native_gt).resize((256,192)).save(original_gt)
    Image.new("RGB",(256,192),(0,0,0)).save(original_render)
    native={"render":str(native_gt),"ground_truth":str(native_gt)}
    direct={"image_name":"view","render":str(original_render),"ground_truth":str(original_gt)}
    row,_=compare_view(native,direct,rgb_metrics)
    assert row["validation"]["decoded_gt_pixels_equal"]
    assert row["delta_psnr_db"]>100
    target=np.asarray(Image.open(original_gt)).copy()
    target[0,0,0]+=1
    Image.fromarray(target).save(original_gt)
    with pytest.raises(ValueError,match="GT differs"):
        compare_view(native,direct,rgb_metrics)


def test_delivery_refuses_incomplete_status(tmp_path):
    import json
    (tmp_path/"benchmark-finalization-status.json").write_text(json.dumps({"phase":"training_priority_ablation"}))
    with pytest.raises(RuntimeError,match="has not completed"):
        validate_completed(tmp_path)


def test_delivery_preserves_metrics_and_arrays_excludes_large_or_private_inputs(tmp_path):
    import json
    import zipfile
    root=tmp_path.resolve()
    models=[root/"benchmark-results"/s/m for s in ["bonsai","teatime"] for m in ["fast","balanced","fine"]]+[root/"teatime-priority/balanced"]
    for model in models:
        views=[]
        for i in range(5):
            paths={}
            for key,folder in [("render","renders"),("ground_truth","gt")]:
                path=model/"test/ours_1"/folder/f"{i:05d}.png"
                path.parent.mkdir(parents=True,exist_ok=True)
                Image.new("RGB",(8,8),(i,i,i)).save(path)
                paths[key]=str(path)
            views.append({"image_name":str(i),**paths,"psnr_db":float(i)})
        (model/"result.json").write_text(json.dumps({"evaluation":{"per_view":views}}))
    for mode in ["fast","balanced","fine","balanced-priority"]:
        path=root/"semantic-results"/mode
        path.mkdir(parents=True)
        np.savez_compressed(path/"semantic_membership.npz",class_0=np.array([True,False,True]))
        (path/"semantic_classes.json").write_text(json.dumps({"classes":["cup"]}))
        Image.new("L",(8,8),255).save(path/"00-cup-masks.png")
        for i in range(5):
            Image.new("RGB",(8,8)).save(path/f"{i:02d}-rgb.png")
    native=root/"full-resolution-results/bonsai/fine"
    native.mkdir(parents=True)
    Image.new("RGB",(8,8)).save(native/"large-gt.png")
    (native/"result.json").write_text(json.dumps({"every_metric":[1,2,3,4,5]}))
    preview=root/"previews/bonsai"
    preview.mkdir(parents=True)
    (preview/"scene.json").write_text(json.dumps({"gaussians":[{"source_index":0},{"source_index":4}]}))
    np.save(preview/"selected_indices.npy",np.array([0,4]),allow_pickle=False)
    (preview/"full_model.ply").write_bytes(b"full PLY must stay remote")
    private=root/"data"
    private.mkdir()
    (private/"raw.png").write_bytes(b"never traversed")
    (models[0]/"credentials.json").write_text('{"private":"never read"}')
    (models[0]/"point_cloud").symlink_to(private,target_is_directory=True)
    included,excluded=choose_files(root)
    assert "full-resolution-results/bonsai/fine/result.json" in included
    assert "full-resolution-results/bonsai/fine/large-gt.png" not in included
    assert "previews/bonsai/scene.json" in included
    assert "previews/bonsai/selected_indices.npy" in included
    assert not any(path.startswith("data/") or path.endswith(".ply") or "credentials" in path for path in included)
    original=next(p for p in included if p.endswith("fast/result.json"))
    assert len(json.loads(included[original]["path"].read_text())["evaluation"]["per_view"])==5
    for model in models:
        prefix=model.relative_to(root).as_posix()
        assert sum(p.startswith(prefix+"/test/ours_1/renders/") for p in included)==3
    assert sum(p.endswith("semantic_membership.npz") for p in included)==4
    manifest=build_manifest(root,included,excluded,{"fixture":True},{"fixture":True})
    archive=root/"fixture.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as zipped:
        for name,record in included.items():
            zipped.write(record["path"],name)
    verify_archive(archive,manifest)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.read("previews/bonsai/selected_indices.npy")== (preview/"selected_indices.npy").read_bytes()

    # The optional full model is copied byte-for-byte only after its actual
    # hash matches the checkpoint recorded by the completed evaluation.
    import hashlib
    full_name="benchmark-results/bonsai/fine/point_cloud/iteration_22000/point_cloud.ply"
    full_path=root/full_name
    full_path.parent.mkdir(parents=True)
    original_ply=b"ply\nformat binary_little_endian 1.0\nend_header\n\x00\xff\x80\x01"
    full_path.write_bytes(original_ply)
    with pytest.raises(ValueError,match="hash differs"):
        choose_files(root,True,"0"*64)
    original_hash=hashlib.sha256(original_ply).hexdigest()
    full_included,full_excluded=choose_files(root,True,original_hash)
    assert [name for name in full_included if name.endswith(".ply")]==[full_name]
    full_manifest=build_manifest(root,full_included,full_excluded,{"fixture":True},{"fixture":True})
    assert full_manifest["full_original_bonsai_fine_ply_included"]
    full_archive=root/"fixture-full.zip"
    with zipfile.ZipFile(full_archive,"w",zipfile.ZIP_DEFLATED) as zipped:
        for name,record in full_included.items():
            zipped.write(record["path"],name)
    verify_archive(full_archive,full_manifest)
    with zipfile.ZipFile(full_archive) as zipped:
        assert zipped.read(full_name)==original_ply
