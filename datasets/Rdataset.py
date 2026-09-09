from torch.utils.data import Dataset
import os
import random
import numpy as np
import torch

metasize = {
    'allegro.pt':26272,
    'barrett.pt':190207,
    'ezgripper.pt':43745,
    'manohand.pt':31292,
    'robotiq_3finger.pt':5863,
    'shadowhand.pt':21917
}

class Rdataset(Dataset):
    def __init__(self, root, category=[], split: str='train',testnum=500, 
                mode = 'cmap', npoints=300, sv_samples=30, scale=1):
        self.split = split
        self.npoints = npoints
        self.sv_samples = sv_samples
        self.root = root
        self.category = category
        self.scale = scale
        self.mode = mode
        self.testnum = testnum

        #expect:xyz+links+class+index?
        self.train_hands = []#xyz+links+dist(class)
        self.train_objs = []#xyz+links+dist(class)
        self.train_sv = []#save point

        self.test_hands = []#xyz+links+dist(class)
        self.test_objs = []#xyz+links+dist(class)
        self.test_sv = []#save point
        
        self.all_hands = []#xyz+links+dist(class)
        self.all_objs = []#xyz+links+dist(class)
        self.all_sv = []#save point

        siz = []
        for i in category:
            siz.append(metasize[i])
        self.minisize = min(siz)

        for i,filename in enumerate(self.category):
            file_path = os.path.join(self.root, filename)
            all_data = torch.load(file_path)
            all_data['hands'][:,:,-1] = i#打标签
            all_data['objs'][:,:,-1] = i

            sam = random.sample(range(all_data['hands'].size(0)),self.minisize)
            all_data['hands'] = all_data['hands'][sam].to(torch.float)
            all_data['objs'] = all_data['objs'][sam].to(torch.float)
            if self.mode == 'cmap':
                self.all_sv.append(all_data['objs'][:,:self.sv_samples])
                self.train_sv.append(all_data['objs'][self.testnum:,:self.sv_samples])
                self.test_sv.append(all_data['objs'][:self.testnum,:self.sv_samples])
            else:
                self.all_sv.append(all_data['hands'][:,:self.sv_samples])
                self.train_sv.append(all_data['hands'][self.testnum:,:self.sv_samples])        
                self.test_sv.append(all_data['hands'][:self.testnum,:self.sv_samples])   
            self.all_hands.append(all_data['hands'])
            self.all_objs.append(all_data['objs'])

            self.train_hands.append(all_data['hands'][self.testnum:])
            self.test_hands.append(all_data['hands'][:self.testnum])

            self.train_objs.append(all_data['objs'][self.testnum:])
            self.test_objs.append(all_data['objs'][:self.testnum])
            
        self.all_hands = torch.cat(self.all_hands, dim=0)
        self.all_objs = torch.cat(self.all_objs, dim=0)
        self.all_sv = torch.cat(self.all_sv, dim=0)

        self.train_hands = torch.cat(self.train_hands, dim=0)
        self.train_objs = torch.cat(self.train_objs, dim=0)
        self.train_sv = torch.cat(self.train_sv, dim=0)

        self.test_hands = torch.cat(self.test_hands, dim=0)
        self.test_objs = torch.cat(self.test_objs, dim=0)
        self.test_sv = torch.cat(self.test_sv, dim=0)

        self.train_hands[:,:self.sv_samples] = self.train_sv
        self.test_hands[:,:self.sv_samples] = self.test_sv
        print('Dataset loaded.')
    def __len__(self):
        if self.split == 'train':
            return self.train_hands.size(0)
        elif self.split == 'test':
            return self.test_hands.size(0)
    def __getitem__(self, index):
        if self.split == 'train':
            sv_points = self.train_sv[index].unsqueeze(0)
            data = torch.cat([
                sv_points,
                torch.zeros(sv_points.shape[0], self.npoints - sv_points.shape[1], sv_points.shape[2])
            ], dim=1)
            masks = torch.zeros_like(data)
            masks[:, :sv_points.shape[1]] = 1
            return {
                'idx':index,
                'train_points': self.train_hands[index],
                'sv_points': data,
                'obj_points': self.train_objs[index],
                'masks': masks,
            }
        else:
            sv_points = self.test_sv[index].unsqueeze(0)
            data = torch.cat([
                sv_points,
                torch.zeros(sv_points.shape[0], self.npoints - sv_points.shape[1], sv_points.shape[2])
            ], dim=1)
            masks = torch.zeros_like(data)
            masks[:, :sv_points.shape[1]] = 1
            return {
                'idx':index,
                'train_points': self.test_hands[index],
                'sv_points': data,
                'obj_points': self.test_objs[index],
                'masks': masks
            }


if __name__ == "__main__":
    tr_dataset = Rdataset(scale=1, 
        root='data_hands',
        category=['shadowhand.pt','manohand.pt'],
        split='test',testnum=500, mode='cmap', npoints=300, sv_samples=30) 
    sam0 = tr_dataset[0]['train_points'].squeeze(0)
    obj0 = tr_dataset[0]['obj_points'].squeeze(0)
    np.save('trash/allegro0.npy',sam0.cpu().numpy())
    np.save('trash/allegro0_obj.npy',obj0.cpu().numpy())
    
        