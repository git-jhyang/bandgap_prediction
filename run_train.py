from model.model_00 import DistNN as DNNZero
from model.model_01r import DistNN as DNNOne
from model.model_02ai import DistNN as DNNTwoAI
from model.model_02ei import DistNN as DNNTwoEI
from model.model_04g import DistNN as DNNG
from util.input_data import Dataset
from main import exec_model_until_success
import util.trainer_log as trl
import util.trainer as tr
import util.trainer_mix as trm
import util.trainer_graph as trg
dataset = Dataset()

map_object = {
    'M00L1_NN':(DNNZero, tr, 'L1'),
    'M01L1_R':(DNNOne, tr, 'L1'),
#    'M02Log_AI':(DNNTwoAI, trl, 'L1log'),
    'M02Log_EI':(DNNTwoEI, trl, 'L1log'),
#    'M02Mix_AI':(DNNTwoAI, trm, 'L1+L1log'),
    'M02Mix_EI':(DNNTwoEI, trm, 'L1+L1log'),
#    'M02L1_AI':(DNNTwoAI, tr, 'L1'),
    'M02L1_EI':(DNNTwoEI, tr, 'L1'),
#    'M04L1_G':(DNNG, trg, 'L1'),
}

#for scale in ['metal_FFF', 'metal_FTT','metal_TFF','metal_TTT']:
#for scale in ['metal_TTT']:
#    dataset.load_dataset(f'c:/WORKSPACE_KRICT/DATA/data_snu/inputdata_{scale}.pickle', True)
#    for mtype, (mobj, trobj, ltyp) in map_object.items():
#        if mtype == 'M00L1_NN':
#            continue
#        for mr in [0.000, 0.010, 0.025, 0.050, 0.100, 0.200, 1.000]:
#            if mtype == 'M01L1_R' and mr < 0.015:
#                continue
#            exec_model_until_success(
#                                kmax=10, model_type=mtype,
#                                dataset=dataset, mobj=mobj, tr=trobj, 
#                                model_name='{}_{}_M{:.3f}'.format(scale, ltyp, mr),
#                                metal_ratio=mr, batch_size=256, num_epochs=200
#                                )
#
#    for mtype, (mobj, trobj, ltyp) in map_object.items():
#        for mr in [0.4, 0.6, 1.0]:
#            exec_model_until_success(
#                                kmax=10, model_type=mtype,
#                                dataset=dataset, mobj=mobj, tr=trobj, 
#                                model_name='{}_{}_M{:.3f}'.format(scale, ltyp, mr),
#                                metal_ratio=mr, batch_size=256, num_epochs=200
#                                )

#for scale in ['metal_FTT','metal_TFF']:
#    dataset.load_dataset(f'c:/WORKSPACE_KRICT/DATA/data_snu/inputdata_{scale}.pickle', True)
#    mtype = 'M04L1_G'
#    mobj, trobj, ltyp = map_object[mtype]
#    for mr in [0.4, 0.6, 1.0]:
#        exec_model_until_success(
#                            kmax=10, model_type=mtype,
#                            dataset=dataset, mobj=mobj, tr=trobj, 
#                            model_name='{}_{}_M{:.3f}'.format(scale, ltyp, mr),
#                            metal_ratio=mr, batch_size=256, num_epochs=200
#                            )

#for scale in ['metal_TTT3']:
#    dataset.load_dataset(f'c:/WORKSPACE_KRICT/DATA/data_snu/inputdata_{scale}.pickle', True)
#    for mtype, (mobj, trobj, ltyp) in map_object.items():
#        for mr in [0.0, 0.2, 0.4, 0.6, 1.0]:
#            exec_model_until_success(
#                            kmax=10, model_type=mtype,
#                            dataset=dataset, mobj=mobj, tr=trobj, 
#                            model_name='{}_{}_M{:.3f}'.format(scale, ltyp, mr),
#                            metal_ratio=mr, batch_size=256, num_epochs=200
#                            )

for scale in ['metal_TFF']:
    dataset.load_dataset(f'c:/WORKSPACE_KRICT/DATA/data_snu/inputdata_{scale}.pickle', True)
    mtype = 'M00L1_NN'
    mobj, trobj, ltyp = map_object[mtype]
    for mr in [1.0]:
        exec_model_until_success(
                            kmax=10, model_type=mtype,
                            dataset=dataset, mobj=mobj, tr=trobj, 
                            model_name='{}_{}_M{:.3f}'.format(scale, ltyp, mr),
                            metal_ratio=mr, batch_size=256, num_epochs=200
                            )
