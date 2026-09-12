"""Bounded priority refinement inside the requested RGB iteration budget."""
PROFILES={
 'fast':{'steps':400,'max_side':1024,'growth_fraction':.10,'max_new_points':3000},
 'balanced':{'steps':1200,'max_side':1536,'growth_fraction':.20,'max_new_points':8000},
 'fine':{'steps':2400,'max_side':2048,'growth_fraction':.25,'max_new_points':16000},
}

def detail_profile(mode,iterations):
    row=dict(PROFILES[mode]);row['start']=iterations-row['steps']+1
    row.update(total_iterations=iterations,interval=100,gradient_weight=.15)
    return row

TRAINING_HOOK=r'''
import copy as splat_copy
import json as splat_detail_json
import base64 as splat_base64
import io as splat_io
import math as splat_math
from PIL import Image as SplatDetailImage
from pathlib import Path as SplatDetailPath
class SplatPriorityDetail:
    def __init__(self,dataset,opt,scene):
        self.config=__DETAIL_CONFIG__;self.mask_dir=SplatDetailPath(__DETAIL_MASK_DIR__)
        self.dataset=dataset;self.scene=scene;self.opt=opt;self.cache={};self.before=[]
        self.added=0;self.limit=None;self.visits={};self.mask_cache={}
        self.folder=SplatDetailPath(scene.model_path).parent/'priority-detail';self.folder.mkdir(exist_ok=True)
    def camera(self,cam,iteration):
        if iteration<self.config['start'] or not (self.mask_dir/(cam.image_name+'.png')).is_file():return cam
        if cam.image_name not in self.cache:
            if len(self.cache)>=4:self.cache.pop(next(iter(self.cache)))
            image_path=SplatDetailPath(self.dataset.source_path)/'images'/cam.image_name
            with SplatDetailImage.open(image_path) as im:
                im=im.convert('RGB');im.thumbnail((self.config['max_side'],self.config['max_side']))
                target=torch.as_tensor(splat_np.array(im).copy(),device='cuda',dtype=torch.float32).permute(2,0,1)/255.
            upgraded=splat_copy.copy(cam)
            upgraded.original_image=target;upgraded.image_width=target.shape[2];upgraded.image_height=target.shape[1]
            upgraded.alpha_mask=torch.ones_like(target[:1]);upgraded.depth_reliable=False
            # Same calibrated frustum and pose, only a denser image grid.
            self.cache[cam.image_name]=upgraded
        return self.cache[cam.image_name]
    def mask(self,cam):
        key=(cam.image_name,cam.image_height,cam.image_width)
        if key not in self.mask_cache:
            if len(self.mask_cache)>=8:self.mask_cache.pop(next(iter(self.mask_cache)))
            path=self.mask_dir/(cam.image_name+'.png')
            if not path.is_file():return None
            with SplatDetailImage.open(path) as im:
                im=im.convert('L').resize((cam.image_width,cam.image_height),SplatDetailImage.Resampling.NEAREST)
                self.mask_cache[key]=torch.as_tensor(splat_np.array(im).copy(),device='cuda')>127
        return self.mask_cache[key]
    def edge_loss(self,image,target,cam,iteration):
        if iteration<self.config['start']:return image.sum()*0
        mask=self.mask(cam)
        if mask is None or not mask.any():return image.sum()*0
        self.visits[cam.image_name]=self.visits.get(cam.image_name,0)+1
        loss=image.sum()*0
        # Only derivatives whose two pixels lie in the reviewed target region.
        for axis in (1,2):
            if axis==1:
                valid=mask[1:]&mask[:-1];diff=(image[:,1:]-image[:,:-1])-(target[:,1:]-target[:,:-1])
            else:
                valid=mask[:,1:]&mask[:,:-1];diff=(image[:,:,1:]-image[:,:,:-1])-(target[:,:,1:]-target[:,:,:-1])
            if valid.any():loss=loss+(diff.abs()*valid).sum()/(3*valid.sum())
        return self.config['gradient_weight']*loss
    def densify(self,gaussians,cam,viewspace,visible,radii,iteration):
        start=self.config['start'];step=iteration-start
        if step<0 or step>=self.config['steps']//2 or step%self.config['interval']:return
        if self.limit is None:self.limit=min(self.config['max_new_points'],int(gaussians.get_xyz.shape[0]*self.config['growth_fraction']))
        remaining=self.limit-self.added
        if remaining<=0 or viewspace.grad is None:return
        mask=self.mask(cam)
        if mask is None:return
        xyz=gaussians.get_xyz;hom=torch.cat([xyz,torch.ones_like(xyz[:,:1])],1)
        clip=hom@cam.full_proj_transform;ndc=clip[:,:2]/clip[:,3:].clamp_min(1e-8)
        pixels=((ndc+1)*torch.tensor([cam.image_width,cam.image_height],device=xyz.device)/2).long()
        # Pinned upstream returns visibility_filter as K x 1 indices, not an
        # N-element bool mask. Use radii to avoid accidental K x N broadcasting.
        valid=(clip[:,3]>0)&(radii>0)&(pixels[:,0]>=0)&(pixels[:,0]<cam.image_width)&(pixels[:,1]>=0)&(pixels[:,1]<cam.image_height)
        ids=torch.where(valid)[0]
        ids=ids[mask[pixels[ids,1],pixels[ids,0]]]
        scores=torch.norm(viewspace.grad[:,:2],dim=1)
        ids=ids[(scores[ids]>1e-9)&(gaussians.get_opacity[ids,0]>.05)]
        take=min(remaining,len(ids),max(1,self.limit//6))
        if not take:return
        chosen=ids[torch.topk(scores[ids],take).indices];grads=torch.zeros((len(xyz),1),device=xyz.device);grads[chosen]=1.
        # Each selected parent adds exactly one net Gaussian via clone or a two-child split.
        gaussians.tmp_radii=radii
        gaussians.densify_and_clone(grads,.5,self.scene.cameras_extent)
        gaussians.densify_and_split(grads,.5,self.scene.cameras_extent)
        self.added+=gaussians.get_xyz.shape[0]-len(xyz)
        gaussians.tmp_radii=None
        if self.added>self.limit:raise RuntimeError('Priority densification exceeded its declared budget')
    def snapshot(self,gaussians,render,pipe,background,phase):
        rows=[]
        for camera in self.scene.getTrainCameras():
            cam=self.camera(camera,self.config['start']);mask=self.mask(cam)
            if mask is None or not mask.any():continue
            with torch.no_grad():
                image=render(cam,gaussians,pipe,background)['render'].clamp(0,1);target=cam.original_image
                mse=((image-target).square()*mask).sum()/(3*mask.sum());l1=((image-target).abs()*mask).sum()/(3*mask.sum())
                coords=torch.where(mask);box=[max(0,int(coords[1].min())-12),max(0,int(coords[0].min())-12),min(cam.image_width,int(coords[1].max())+13),min(cam.image_height,int(coords[0].max())+13)]
                rgb=(image.permute(1,2,0)*255).byte().cpu().numpy();im=SplatDetailImage.fromarray(rgb).crop(box)
                im.thumbnail((640,640));buffer=splat_io.BytesIO();im.save(buffer,format='JPEG',quality=92)
                rows.append({'image_name':cam.image_name,'width':cam.image_width,'height':cam.image_height,'box':box,
                    'roi_psnr':float(-10*torch.log10(mse.clamp_min(1e-12))),'roi_l1':float(l1),
                    'image':'data:image/jpeg;base64,'+splat_base64.b64encode(buffer.getvalue()).decode()})
            if len(rows)==3:break
        return rows
    def begin(self,gaussians,render,pipe,background,iteration):
        if iteration!=self.config['start']:return
        self.initial_count=int(gaussians.get_xyz.shape[0]);self.scene.save(iteration-1)
        self.before=self.snapshot(gaussians,render,pipe,background,'before')
        print('PRIORITY_DETAIL_START',iteration,self.config,flush=True)
    def finish(self,gaussians,render,pipe,background):
        after=self.snapshot(gaussians,render,pipe,background,'after')
        evidence={'status':'completed','method':'reviewed_mask_weighting_high_resolution_gradient_and_bounded_roi_densification',
            'profile':self.config,'before_iteration':self.config['start']-1,'after_iteration':self.config['total_iterations'],
            'added_gaussians':self.added,'growth_limit':self.limit or 0,'before_gaussians':self.initial_count,
            'after_gaussians':int(gaussians.get_xyz.shape[0]),'high_resolution_visits':self.visits,
            'before':self.before,'after':after,'comparison_scope':'same training cameras before and after final refinement; not a held-out evaluation or unweighted ablation'}
        (self.folder/'evidence.json').write_text(splat_detail_json.dumps(evidence,ensure_ascii=False,allow_nan=False))
        print('PRIORITY_DETAIL_COMPLETED',self.added,self.visits,flush=True)
'''

def patch_detail(source,profile,mask_dir):
    def replace(marker,replacement):
        nonlocal source
        if source.count(marker)!=1:raise ValueError('上游重点增强适配位置已变化：'+marker[:70])
        source=source.replace(marker,replacement)
    replace('    ema_loss_for_log = 0.0','    priority_detail = SplatPriorityDetail(dataset,opt,scene)\n    ema_loss_for_log = 0.0')
    replace('        # Render','        priority_detail.begin(gaussians,render,pipe,background,iteration)\n        viewpoint_cam = priority_detail.camera(viewpoint_cam,iteration)\n\n        # Render')
    replace('        loss.backward()','        loss = loss + priority_detail.edge_loss(image,gt_image,viewpoint_cam,iteration)\n        loss.backward()')
    replace('            # Densification','            priority_detail.densify(gaussians,viewpoint_cam,viewspace_point_tensor,visibility_filter,radii,iteration)\n            # Densification')
    replace('def prepare_output_and_logger(args):','    priority_detail.finish(gaussians,render,pipe,background)\n\ndef prepare_output_and_logger(args):')
    hook=TRAINING_HOOK.replace('__DETAIL_CONFIG__',repr(profile)).replace('__DETAIL_MASK_DIR__',repr(str(mask_dir)))
    return source.replace('def training(',hook+'\ndef training(',1)
