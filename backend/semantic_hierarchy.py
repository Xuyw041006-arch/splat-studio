"""Generate explicit scene/object/part hypotheses before geometric verification.

The vocabulary proposes relations; repeated visible containment is still required
by semantic_granularity. A missing part detection never invents a part or a mask.
"""
from collections import defaultdict
import numpy as np
from .semantic_targets import normalize_label, _read_mask, _frame

# child -> (parent, relationship). These are hypotheses, not detector ground truth.
RELATIONS = {
    'bear nose': ('stuffed bear', 'part_of'), 'bear ear': ('stuffed bear', 'part_of'),
    'mug handle': ('coffee mug', 'part_of'), 'coffee': ('coffee mug', 'contents_of'),
    'chair back': ('chair', 'part_of'), 'chair leg': ('chair', 'part_of'),
    'door handle': ('door', 'part_of'), 'car window': ('car', 'part_of'),
    'wheel': ('car', 'part_of'), 'bottle cap': ('bottle', 'part_of'),
    'lamp shade': ('lamp', 'part_of'), 'table leg': ('table', 'part_of'),
}


def hierarchy_vocabulary(labels):
    labels=list(dict.fromkeys(labels))
    present={normalize_label(x) for x in labels}
    return labels+[child for child,(parent,_) in RELATIONS.items()
                   if parent in present and child not in present]


def prepare_hierarchy_records(records):
    """Scene collection contains observed objects; named parts need mask evidence.

    Called after masks have been resized to calibrated training grids. Coarse
    masks are the observed object union, never an all-image or unseen-region mask.
    Native s/m/l imports retain their explicit declarations unchanged.
    """
    output=[];frames=defaultdict(list)
    for i,original in enumerate(records):
        row=dict(original)
        row.setdefault('region_id',row.get('mask_id',f'region-{i}'))
        if row.get('granularity',row.get('level')) in {'s','m','l','coarse'}:
            output.append(row);continue
        row['label']=normalize_label(row['label'])
        row.setdefault('level',row.get('granularity','part' if row['label'] in RELATIONS else 'object'))
        if row['label'] in RELATIONS:
            row['level']='part'
            if 'granularity' in row:row['granularity']='part'
        frames[_frame(row)].append(row)
    for frame,rows in sorted(frames.items()):
        masks={row['region_id']:_read_mask(row,None) for row in rows}
        nonempty=[row for row in rows if masks[row['region_id']].any()]
        if not nonempty:
            output.extend(rows);continue
        shapes={masks[r['region_id']].shape for r in rows}
        if len(shapes)!=1:raise ValueError('Hierarchy masks need a shared calibrated pixel grid')
        union=np.logical_or.reduce([masks[row['region_id']] for row in nonempty])
        group_id='scene-observed-collection'
        if any(r['region_id']==group_id for r in rows):raise ValueError('Reserved scene hierarchy region ID')
        group={'frame_id':frame,'region_id':group_id,'mask_id':group_id,
               'label':'scene objects','level':'coarse','mask':union,
               'confidence':float(np.mean([r.get('confidence',1.) for r in nonempty])),
               'annotation_source':'observed_object_union','semantic_name_known':True}
        for row in rows:
            if row.get('parent_region_id') or row.get('parent_id'):continue
            relation=RELATIONS.get(row['label'])
            if relation:
                child=masks[row['region_id']];area=int(child.sum())
                parents=[r for r in nonempty if r['label']==relation[0]
                         and area and (child&masks[r['region_id']]).sum()/area>=.9]
                # More than one containing instance stays ambiguous.
                if len(parents)==1:
                    row.update(parent_region_id=parents[0]['region_id'],relation=relation[1],
                               relation_source='semantic_vocabulary_hypothesis')
            elif row.get('level')=='object':
                row.update(parent_region_id=group_id,relation='member_of',
                           relation_source='observed_scene_collection')
        output.extend(rows);output.append(group)
    return output
