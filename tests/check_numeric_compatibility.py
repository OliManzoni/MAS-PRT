"""Synthetic numerical check; isolates scoring functions from GPU imports."""
import ast, json, math
from pathlib import Path
from statistics import mean
import numpy as np
import pandas as pd
from shapely import Polygon, Point
source = Path(__file__).resolve().parents[1] / "MAS/functions.py"
tree = ast.parse(source.read_text())
names = {"dist", "mean_calculator", "remove_outliers", "dam_above", "first_encounter", "dam_distance", "is_nest_poly"}
module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
ns = dict(pd=pd, np=np, m=math, mean=mean, Point=Point)
exec(compile(module, str(source), "exec"), ns)
a = np.zeros((6,40))
for frame in range(6):
    a[frame,0] = frame
    for col in range(1,40,3):
        a[frame,col:col+3] = [100+20*frame+col, 100+col, 1]
df, size = ns["mean_calculator"](pd.DataFrame(a))
assert size > 0
assert ns["first_encounter"](df) == 1
df.loc[2,["pupx","pupy"]] = -1
ns["remove_outliers"](df)
assert df.loc[2,"pupx"] == (df.loc[1,"pupx"]+df.loc[3,"pupx"])/2
# Each of the five movements is 20 pixels, outside the remote nest.
nest = Polygon([(0,0),(1,0),(1,1),(0,1)])
assert ns["dam_distance"](df,nest,stop=6) == 100
print(json.dumps({"pandas":pd.__version__,"numpy":np.__version__,"size":size,"coordinates":df.astype(float).values.tolist()}))
