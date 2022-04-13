from Networks import Model

data_dir = '/home/chentyt/Documents/4tb/Tiana/P100ObjDet/Data/RCA_Split1/'
fname = ['train_v1', 'test_v1', 'valid_v1']

seed = 17322625
epoch_no = 500
patience = 100
lr = 1e-6
batch_size = 8

# Naming
date = '2022-04-12'
backbone_name = 'Resnet_simple'
save_weight_metric = 'F1Val'
roi_bbatch = 64

# Faster RCNN
rpn_nms_threshold = 0.3  # TODO #default:0.7
model = Model(nms_thresh=rpn_nms_threshold)

hyp = f'boxBatchSize:{roi_bbatch},lr{lr}'
model_id = f'{date}_{backbone_name}_{save_weight_metric}_rpnNms{float(rpn_nms_threshold)}_{hyp}'

output_dir = "/home/chentyt/Documents/4tb/Tiana/P100ObjDet/Model/"
log_dir = f'{output_dir}{model_id}_log'
model_dir = f'{output_dir}{model_id}_weight'
