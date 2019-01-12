#!/usr/bin/env python3
#Power Diagramer
#Author: Richard Barnes (rbarnes.org)
#!/usr/bin/env python3
import plumbum
import random
import shapely
import shapely.wkt
from matplotlib import pyplot as plt
from descartes import PolygonPatch

def GetPowerDiagram(points, ray_length=1000, crop=True):
  """Generates a power diagram of a set of points.

  Arguments:

    points - A list of points of the form `[(x,y,weight), (x,y,weight), ...]`

    ray_length - The power diagram contains infinite rays. The direction vector
                 of those rays will be multiplied by `ray_length` and the ends
                 of the rays connected in order to form a finite representation
                 of the polygon

    crop       - If `True`, then the bounded representation above is cropped to
                 the bounding box of the point cloud
  """
  powerd = plumbum.local["./power_diagramer.exe"]

  #Format output for reading by power_diagramer.exe
  points = [map(str,x) for x in points]
  points = [' '.join(x) for x in points]
  points = '\n'.join(points)
  points = '{raylen}\n{crop}\n{points}'.format(
    raylen = ray_length,
    crop   = 'CROP' if crop else 'NOCROP',
    points = points
  )

  #Run the command
  polygons = (powerd["-"] << points)()

  #Get the output of `power_diagramer.exe`. It is in WKT format, one polygon per
  #line.
  polygons = polygons.split("\n")
  polygons = [x.strip() for x in polygons]
  polygons = [x for x in polygons if len(x)>0]
  polygons = [shapely.wkt.loads(x) for x in polygons]

  #Generate bounding box for ease in plotting
  bbox = [x.bounds for x in polygons]
  minx = min([x[0] for x in bbox])
  miny = min([x[1] for x in bbox])
  maxx = max([x[2] for x in bbox])
  maxy = max([x[3] for x in bbox])

  return polygons, (minx,miny,maxx,maxy)

POINT_COUNT = 100
pts         = []
for i in range(POINT_COUNT):
  x      = random.uniform(0,100) 
  y      = random.uniform(0,100)
  weight = random.uniform(0,10)
  pts.append((x,y,weight))

polys, (minx, miny, maxx, maxy) = GetPowerDiagram(pts, ray_length=1000, crop=True)

fig = plt.figure(1, figsize=(5,5), dpi=90)
ax = fig.add_subplot(111)
ax.set_xlim(minx,maxx)
ax.set_ylim(miny,maxy)
for poly in polys:
  ax.add_patch(PolygonPatch(poly))
plt.show()