import os
import re
import numpy as np
import glob

numShape = 11

def processMesh(mesh):
    reComp = re.compile("(?<=^)(v |vn |vt |f )(.*)(?=$)", re.MULTILINE)
    with open(mesh) as f:
        data = [txt.group() for txt in reComp.finditer(f.read())]
    v_arr, f_arr = [], []
    for line in data:
        tokens = line.split(' ')
        if tokens[0] == 'v':
            v_arr.append([float(c) for c in tokens[1:]])
        elif tokens[0] == 'f':
            #f_arr.append([int(c) for c in tokens[1:]])
            f_arr.append([int(c) for c in tokens[1:]])
    return v_arr, f_arr

for idx in range(numShape):
    #modelS_dir = 'data/ShapeNet%d/modelS' % (idx+1)
    #modelT_dir = 'data/ShapeNet%d/modelT' % (idx+1)
    modelS_dir = glob.glob('data/ShapeNet%d_*/modelS' % (idx+1))[0]
    modelT_dir = glob.glob('data/ShapeNet%d_*/modelT' % (idx+1))[0]

    print('Processing %s ...' % modelS_dir)
    vS, fS = processMesh('%s/model.obj' % (modelS_dir))
    np.savetxt('%s/mesh.vert' % (modelS_dir), vS, fmt = '%.5f')
    np.savetxt('%s/mesh.triv' % (modelS_dir), fS, fmt = '%d')
    print('Processing %s ...' % modelT_dir)
    vT, fT = processMesh('%s/model.obj' % (modelT_dir))
    np.savetxt('%s/mesh.vert' % (modelT_dir), vT, fmt = '%.5f')
    np.savetxt('%s/mesh.triv' % (modelT_dir), fT, fmt = '%d')


