import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from metrics.eval_reid import *
from data.triplet_sampler import *
from typing import OrderedDict
from processor import get_model, embed_for_retrieval, compute_distmat
import torch.multiprocessing
import os
import yaml
from utils import re_ranking, graph_reranking, GCNRefiner
#import cv2



def normalize_batch(batch, maximo=None, minimo = None):
    if maximo != None:
        return (batch - minimo.unsqueeze(-1).unsqueeze(-1)) / (maximo.unsqueeze(-1).unsqueeze(-1) - minimo.unsqueeze(-1).unsqueeze(-1))
    else:
        return (batch - torch.amin(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1)) / (torch.amax(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1) - torch.amin(batch, dim=(1, 2)).unsqueeze(-1).unsqueeze(-1))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic= True
    torch.backends.cudnn.benchmark= False

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

### Save activations may be not working properly after changing the code
def save_activ(activations, count_imgs, data, transform, path_names, blend_ratio, q_or_g=""):
            cnt = 0
            for item in activations:
                if item.ndim == 4:
                    mapactiv = item.sum(dim=1) 
                    mapactiv = normalize_batch(mapactiv, None, None) 
                    mapactiv = transform(mapactiv).cpu().numpy()                    
                    for i in range(mapactiv.shape[0]):
                        if data['dataset'] == 'VERIWILD':
                            path2img = data['ROOT_DIR'] + path_names[count_imgs + i, 0]
                            outputDIR = args.path_weights + '/activations'+q_or_g+'/'
                            if not os.path.exists(outputDIR): os.mkdir(outputDIR)
                        else:
                            path2img = data['teste_dir'] + str(path_names[count_imgs + i])
                            outputDIR = args.path_weights + '/activations'+q_or_g+'/'
                            if not os.path.exists(outputDIR): os.mkdir(outputDIR)
                        img_og = cv2.resize(cv2.imread(path2img), (256,256), interpolation= cv2.INTER_LINEAR)
                        if cnt == 0:            
                            activ = cv2.applyColorMap(np.uint8(mapactiv[i,:,:]* 255), cv2.COLORMAP_JET)
                            activ = np.uint8(img_og * blend_ratio + activ[:,:,:3] *(1-blend_ratio))
                            if not os.path.exists(outputDIR + 'cross_entropy_branch/'): os.mkdir(outputDIR + 'cross_entropy_branch/')
                            cv2.imwrite(outputDIR + 'cross_entropy_branch/' +str(count_imgs + i) + '.jpg', activ)
                        if cnt == 1:            
                            activ = cv2.applyColorMap(np.uint8(mapactiv[i,:,:]* 255), cv2.COLORMAP_JET)
                            activ = np.uint8(img_og * blend_ratio + activ[:,:,:3] *(1-blend_ratio))
                            if not os.path.exists(outputDIR + 'triplet_branch/'): os.mkdir(outputDIR + 'triplet_branch/')
                            cv2.imwrite(outputDIR + 'triplet_branch/' +str(count_imgs + i) + '.jpg', activ)
                        if cnt == 2:            
                            activ = cv2.applyColorMap(np.uint8(mapactiv[i,:,:]* 255), cv2.COLORMAP_JET)
                            activ = np.uint8(img_og * blend_ratio + activ[:,:,:3] *(1-blend_ratio))
                            if not os.path.exists(outputDIR + 'mhsa_ce_branch/'): os.mkdir(outputDIR + 'mhsa_ce_branch/')
                            cv2.imwrite(outputDIR + 'mhsa_ce_branch/' +str(count_imgs + i) + '.jpg', activ)
                        if cnt == 3:            
                            activ = cv2.applyColorMap(np.uint8(mapactiv[i,:,:]* 255), cv2.COLORMAP_JET)
                            activ = np.uint8(img_og * blend_ratio + activ[:,:,:3] *(1-blend_ratio))
                            if not os.path.exists(outputDIR + 'mhsa_t_branch/'): os.mkdir(outputDIR + 'mhsa_t_branch/')
                            cv2.imwrite(outputDIR + 'mhsa_t_branch/' +str(count_imgs + i) + '.jpg', activ)
                    cnt += 1

def test_epoch(model, device, dataloader_q, dataloader_g, model_arch, remove_junk=True, scaler=None, re_rank=False, graph_re_rank=False, learn_based=False, gcn_model=None, use_camera=True):
    """re_rank only           -> k-reciprocal
       re_rank + graph_re_rank -> rule-based graph re-ranking
       + learn_based (+ gcn)   -> LiteGCR on the full query+gallery graph"""
    use_graph = bool(re_rank and graph_re_rank)
    gcn = gcn_model if (use_graph and learn_based) else None
    if gcn is not None:
        gcn.eval()
    model.eval()
    device_type = "cuda" if device.type == "cuda" else "cpu"

    def extract(loader, desc):
        feats, vids, cams = [], [], []
        with torch.no_grad():
            for image, vid, cam_id, view_id in tqdm(loader, desc=desc, bar_format='{l_bar}{bar:20}{r_bar}'):
                image = image.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=bool(scaler)):
                    _, _, ffs, _ = model(image, cam_id, view_id)
                feats.append(embed_for_retrieval(ffs, keep_global_raw=gcn is not None).cpu())
                vids.append(torch.as_tensor(vid))
                cams.append(torch.as_tensor(cam_id))
        return torch.cat(feats, 0), torch.cat(vids, 0).numpy(), torch.cat(cams, 0).numpy()

    qf, q_vids, q_camids = extract(dataloader_q, 'Query infer (%)')
    gf, g_vids, g_camids = extract(dataloader_g, 'Gallery infer (%)')

    distmat = compute_distmat(qf, gf, q_camids, g_camids, gcn_model=gcn,
                              re_rank=re_rank and not graph_re_rank, graph_re_rank=use_graph,
                              use_camera=use_camera)
    del qf, gf
    cmc, mAP = eval_func(distmat, q_vids, g_vids, q_camids, g_camids, remove_junk=remove_junk)
    return cmc, mAP


if __name__ == "__main__":

    ### Just to ensure VehicleID 10-fold validation randomness is not random to compare different models training
    set_seed(0)
    parser = argparse.ArgumentParser(description='Reid train')

    parser.add_argument('--batch_size', default=None, type=int, help='an integer for the accumulator')
    parser.add_argument('--dataset', default=None, help='Choose one of[Veri776, VERIWILD]')
    parser.add_argument('--model_arch', default=None, help='Model Architecture')
    parser.add_argument('--path_weights', default=None, help="Path to *.pth/*.pt loading weights file")
    parser.add_argument('--re_rank', action="store_true", help="Re-Rank")
    parser.add_argument('--graph_re_rank', action="store_true", help='Graph Re-Rank')
    parser.add_argument('--learn_based', action="store_true", help="Learn_based graph re-rank")
    args = parser.parse_args()

    with open(args.path_weights + "config.yaml", "r") as stream:
        data = yaml.safe_load(stream)

    data['BATCH_SIZE'] = args.batch_size or data['BATCH_SIZE']
    data['dataset'] = args.dataset or data['dataset']
    data['model_arch'] = args.model_arch or data['model_arch']


    teste_transform = transforms.Compose([
                    transforms.Resize((data['y_length'],data['x_length']), antialias=True),
                    transforms.Normalize(data['n_mean'], data['n_std']),

    ])                  

    if data['half_precision']:
        scaler = torch.cuda.amp.GradScaler()
        print("WARNING: half_precision=True, forcing to False for inference stability")
        data['half_precision'] = False  # Force FP32 for inference
        scaler = False
    else:
        scaler=False

    ### Replace paths as needed
    if data['dataset']== 'VERIWILD':
        data['n_classes'] = 30671
        data_q = CustomDataSet4VERIWILD('/home/eurico/VERI-Wild/train_test_split/test_3000_id_query.txt', data['ROOT_DIR'], transform=teste_transform, with_view=True)
        data_g = CustomDataSet4VERIWILD('/home/eurico/VERI-Wild/train_test_split/test_3000_id.txt', data['ROOT_DIR'], transform=teste_transform, with_view=True)
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste']) #data['BATCH_SIZE']
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])

    if data['dataset']== 'VERIWILD2.0':
        data['n_classes'] = 30671
        vw2_dir = "/mnt/DATADISK/Datasets/vehicle/VeriWild/v2.0/"
        set = 'B' #args.vw2_set A, B or All
        data_q = CustomDataSet4VERIWILDv2(vw2_dir + 'test_split_V2/'+ set +'_query.txt', vw2_dir, transform=teste_transform, with_view=True)
        data_g = CustomDataSet4VERIWILDv2(vw2_dir + 'test_split_V2/'+ set +'_gallery.txt', vw2_dir, transform=teste_transform, with_view=True)
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=0) #data['BATCH_SIZE'] data['num_workers_teste']
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=0)


    if data['dataset'] == 'Veri776':
        data_q = CustomDataSet4Veri776_withviewpont(data['query_list_file'], data['query_dir'], data['train_keypoint'], data['test_keypoint'], is_train=False, transform=teste_transform)
        data_g = CustomDataSet4Veri776_withviewpont(data['gallery_list_file'], data['teste_dir'], data['train_keypoint'], data['test_keypoint'], is_train=False, transform=teste_transform)
        data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
        data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])

    path_weights = args.path_weights + 'best_CMC.pt'
    # GCN weights saved together with best_CMC.pt (fallback: last epoch)
    gcn_path_weights = args.path_weights + 'gcn_best_CMC.pt'
    if not os.path.exists(gcn_path_weights):
        gcn_path_weights = args.path_weights + 'gcn_model.pt'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Create model on CPU first
    model = get_model(data, torch.device("cpu"))
    # Load checkpoint BEFORE moving to GPU
    model.load_state_dict(torch.load(path_weights, map_location='cpu'))
    # Then move to GPU and set eval
    model = model.to(device)
    model.eval()
    
    gcn_model = None
    if args.learn_based:
        # Get length feature
        input = torch.randn((32,3,256,256)).to(device)
        preds, embs, ffs, output = model(input, torch.randint(0,19,(32,1)).long().to(device), torch.randint(0,7,(32,8)).long().to(device))
        feature_dim = (ffs[0] if isinstance(ffs, (list, tuple)) else ffs).shape[1]
        # Create GCN on CPU, load, then move to GPU
        gcn_model = GCNRefiner(feature_dim=feature_dim)
        gcn_model.load_state_dict(torch.load(gcn_path_weights, map_location='cpu'))
        gcn_model = gcn_model.to(device)
        gcn_model.eval()

    mean = False
    l2 = True

    if data['dataset'] == "VehicleID":
        list_mAP = []
        list_cmc1 = []
        list_cmc5 = []
        list_cmc10 = []
        for i in range(10):
            reader = open(os.path.join(os.path.dirname(os.path.normpath(data['ROOT_DIR'])), 'train_test_split', 'test_list_800.txt'))
            lines = reader.readlines()
            random.shuffle(lines)
            data_q = CustomDataSet4VehicleID_Random(lines, data['ROOT_DIR'], is_train=False, mode="q", transform=teste_transform, teste=True)
            data_g = CustomDataSet4VehicleID_Random(lines, data['ROOT_DIR'], is_train=False, mode="g", transform=teste_transform, teste=True)
            data_q = DataLoader(data_q, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
            data_g = DataLoader(data_g, batch_size=data['BATCH_SIZE'], shuffle=False, num_workers=data['num_workers_teste'])
            # VehicleID has no camera labels: LiteGCR cross-camera branch disabled
            cmc, mAP = test_epoch(model, device, data_q, data_g, data['model_arch'], remove_junk=True, scaler=scaler, re_rank=args.re_rank, graph_re_rank=args.graph_re_rank, learn_based=args.learn_based, gcn_model=gcn_model, use_camera=False)
            list_mAP.append(mAP)
            list_cmc1.append(cmc[0])
            list_cmc5.append(cmc[4])
            list_cmc10.append(cmc[9])
        mAP = sum(list_mAP) / len(list_mAP)
        cmc1 = sum(list_cmc1) / len(list_cmc1)
        cmc5 = sum(list_cmc5) / len(list_cmc5)
        cmc10 = sum(list_cmc10) / len(list_cmc10)
        print(f'\n\nmAP = {mAP},  CMC1= {cmc1}, CMC5= {cmc5}, CMC10= {cmc10}')
        with open(args.path_weights +'result_map_l2_'+ str(l2) + '_mean_' + str(mean) +'.npy', 'wb') as f:
            np.save(f, mAP)
        with open(args.path_weights +'result_cmc_l2_'+ str(l2) + '_mean_' + str(mean) +'.npy', 'wb') as f:
            np.save(f, cmc1)
    else:
        if args.re_rank:
            if args.graph_re_rank:
                if args.learn_based:
                    print("Learn-based graph re-ranking")
                    cmc, mAP = test_epoch(model, device, data_q, data_g, data['model_arch'], remove_junk=True, scaler=scaler, re_rank=args.re_rank, graph_re_rank=args.graph_re_rank, learn_based=args.learn_based, gcn_model=gcn_model)
                else:
                    print("Graph re-ranking")
                    cmc, mAP = test_epoch(model, device, data_q, data_g, data['model_arch'], remove_junk=True, scaler=scaler, re_rank=args.re_rank, graph_re_rank=args.graph_re_rank)
            else:
                print("Re-ranking")
                cmc, mAP = test_epoch(model, device, data_q, data_g, data['model_arch'], remove_junk=True, scaler=scaler, re_rank=args.re_rank)
        else:
            print("No Re-ranking")
            cmc, mAP = test_epoch(model, device, data_q, data_g, data['model_arch'], remove_junk=True, scaler=scaler)
        # print(cmc)
        print(f'mAP = {mAP},  CMC1= {cmc[0]}, CMC5= {cmc[4]}, CMC10= {cmc[9]}')
        with open(args.path_weights +'result_map_l2_'+ str(l2) + '_mean_' + str(mean) +'.npy', 'wb') as f:
            np.save(f, mAP)
        with open(args.path_weights +'result_cmc_l2_'+ str(l2) + '_mean_' + str(mean) +'.npy', 'wb') as f:
            np.save(f, cmc)

    print('Weights: ', path_weights)
