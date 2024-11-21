import pygmo
import json    
def _hv_prepare(_pareto_temp):
    for i in range(len(_pareto_temp)):
        for j in range(len(_pareto_temp[0])):
            if(_pareto_temp[i][j]>0):
                _pareto_temp[i][j] *= -1
            else:
                _pareto_temp[i][j] = -0.00000000000000001
    return _pareto_temp
p = json.load(open("data_test2/present/pareto.json"))

front = _hv_prepare(p['front'])

hv = pygmo.hypervolume(front)
print(hv.contributions([0,0,0]))