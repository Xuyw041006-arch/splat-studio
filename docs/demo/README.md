# Splat Studio — screenshot walkthrough

**English** · [简体中文](README.zh-CN.md) · [Project home](../../README.md)

24 scenes captured from the native macOS App, with Chinese and English originals and presentation cards. The real Teatime Fast run uses 177 photos on an A100, exporting **1,200,686 Gaussians with SH3**.

![Actual Teatime reconstruction](screenshots/16-teatime-viewer-en.jpg)

Browse the complete walkthrough below directly on GitHub. To use the interactive bilingual version, download the repository ZIP, keep `docs/demo/` intact, then open **图文展示.html**. **项目报告.html** provides the bilingual system and execution report; [the Markdown report](方法与实测说明.md) can be read here.

The six- and two-photo branches demonstrate input routing, annotation and task preparation. The executed training result uses 177 photos. Fast semantics has partial coverage; the editor examples show actual regions, and the [execution record](execution-record.json) defines the measured scope.

## 01. Two ways to begin

Create a scene from photos, or import an existing reconstruction. The home screen keeps both paths easy to find.

![Two ways to begin](screenshots/01-home-en.jpg)

## 02. Name the scene. Add photos.

This walkthrough uses the Teatime image set. The scene name carries through to saving and export.

![Name the scene. Add photos.](screenshots/02-name-en.jpg)

## 03. Let the photos guide the next step

The App checks 177 photos, finds 177 unique images and reports 97% photo connectivity. It recommends standard reconstruction. Connectivity is an overlap precheck; camera poses are solved during reconstruction.

![Let the photos guide the next step](screenshots/03-analysis-en.jpg)

## 04. Six photos trigger the sparse workflow

With six real Teatime photos, the App detects overlap and recommends sparse reconstruction with optional completion. This verifies input routing; the newly trained showcase model uses all 177 photos.

![Six photos trigger the sparse workflow](screenshots/22-sparse-six-en.jpg)

## 05. Two views prompt completion

With two photos, the App recommends sparse reconstruction with completion and explains that unseen backsides may remain missing. This captures the automatic recommendation and choice, not a verified recovery of hidden geometry.

![Two views prompt completion](screenshots/27-two-view-completion-en.jpg)

## 06. Bring photos from your own camera

Optionally provide a known coverage angle and calibrated intrinsics matched to the photos. Camera poses are solved from the images; count and connectivity checks do not establish precise shooting angles.

![Bring photos from your own camera](screenshots/23-capture-information-en.jpg)

## 07. Add object understanding when needed

Choose appearance-only reconstruction for visual exploration, or enable semantics for finding and editing objects.

![Add object understanding when needed](screenshots/04-semantics-en.jpg)

## 08. Tell the App what matters

This demo requests the bear and coffee mug. Their names are matched to real image regions, which are reviewed before priority training. Switching languages preserves the original user input.

![Tell the App what matters](screenshots/05-priority-en.jpg)

## 09. Mark the object directly in a photo

In the six-photo example, 35 vertices outline the visible upper body of the bear, and the mask is saved. A name, cross-view ID and level organize the annotation; matching the same instance still requires other views.

![Mark the object directly in a photo](screenshots/24-manual-region-en.jpg)

## 10. Review the region before priority training

The green overlay is the manually saved region. Confirmation includes it in the task. Its 100% score is a manual-annotation field, not measured accuracy. The separate A100 run reviewed 28 candidates and retained 11.

![Review the region before priority training](screenshots/25-priority-review-en.jpg)

## 11. Choose quality and compute

This run uses Fast mode on a Colab A100: 7,000 RGB iterations, a base semantic budget of 400 steps over up to 12 sampled views, and the final 400 RGB steps reserved for priority detail.

![Choose quality and compute](screenshots/06-fast-a100-en.jpg)

## 12. Review the plan before starting

The confirmation screen summarizes the scene, photo count, mode, device and estimated duration. It estimates 8–27 minutes here; actual execution time is recorded separately.

![Review the plan before starting](screenshots/07-estimate-en.jpg)

## 13. Use Colab without owning a GPU server

Built-in instructions explain Colab setup and task/result file exchange. This demo executes in a real A100 environment.

![Use Colab without owning a GPU server](screenshots/08-cloud-guide-en.jpg)

## 14. Local NVIDIA setup is documented

Users can open the Windows / Linux CUDA setup instructions inside the App. This screenshot demonstrates the built-in guide.

![Local NVIDIA setup is documented](screenshots/09-local-guide-en.jpg)

## 15. Bring your own cloud workflow

Other GPU servers use the same task-package workflow. Dependency setup, task execution and result import are explained in the App.

![Bring your own cloud workflow](screenshots/10-server-guide-en.jpg)

## 16. Transfer photos and settings to the cloud

The six-photo example exports a task ZIP with its confirmed priority region. Run it using the included instructions, then import the result. This example stops at preparation; the App explicitly shows that training has not started.

![Transfer photos and settings to the cloud](screenshots/26-cloud-package-en.jpg)

## 17. Open an existing result

Choose whether the imported scene includes object semantics. Appearance-only models support 3D exploration; semantic scenes also support object-based interaction.

![Open an existing result](screenshots/14-import-type-en.jpg)

## 18. Pair geometry with its semantics

This demo imports the trained point_cloud.ply and its matching semantics.zip. The App validates their correspondence before opening the complete scene.

![Pair geometry with its semantics](screenshots/15-import-files-en.jpg)

## 19. Explore the trained Teatime scene

A real Fast run on A100 turns 177 photos into 1,200,686 Gaussians with degree-three spherical harmonics. Core-region filtering changes visibility only. Drag to orbit, scroll to zoom, and right-drag to pan.

![Explore the trained Teatime scene](screenshots/16-teatime-viewer-en.jpg)

## 20. Search objects and inspect confidence

Searching for table filters the semantic catalog. This selected region contains 14,543 Gaussians with 52% cross-view confidence. The score is a matching weight; this region covers part of the table.

![Search objects and inspect confidence](screenshots/17-semantic-search-en.jpg)

## 21. Hide a region and keep the source

Hiding the selected semantic region removes its Gaussians from view. Undo restoration was verified. Hiding retains the source file and does not automatically fill exposed gaps.

![Hide a region and keep the source](screenshots/20-hide-region-en.jpg)

## 22. Restore the scene with a command

Typing undo runs a local text command and restores the isolated scene. Text-based undo was verified; the command bar also supports finding, focusing and hiding regions.

![Restore the scene with a command](screenshots/19-command-undo-en.jpg)

## 23. Isolate a region to inspect coverage

Isolate the selected table region to inspect its assigned Gaussians. Coverage in this Fast result is incomplete; this measured view demonstrates region isolation.

![Isolate a region to inspect coverage](screenshots/18-isolate-region-en.jpg)

## 24. Save a complete, reusable scene

The native App exported a complete .splat.jsonl with all Gaussians, spherical harmonics, available semantics and visibility settings. The file is named Teatime-Fast here. System dialog buttons follow the macOS language.

![Save a complete, reusable scene](screenshots/21-save-result-en.jpg)

The `cards/` folder contains 48 captioned images for presentations; `screenshots/` retains 48 untouched original captures. Full training artifacts are stored separately and are not included in this documentation directory.
