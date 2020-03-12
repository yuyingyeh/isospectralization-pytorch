import trimesh as trm

sp = trm.creation.icosphere(radius = 0.45, subdivisions=3)
sp.export('sphere_subd_3.obj') # 642 vertex, 1280 faces

sp = trm.creation.icosphere(radius = 0.45, subdivisions=4)
sp.export('sphere_subd_4.obj') # 2562 vertex, 5120 faces

sp = trm.creation.icosphere(radius = 0.45, subdivisions=5)
sp.export('sphere_subd_5.obj') # 10242 vertex, 20480 faces
