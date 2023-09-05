import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('-o', '--output',    default=None)
parser.add_argument('--dir_num', type=int, default=None)
parser.add_argument('--configs', type=int, default=50)
parser.add_argument('--total_frames', type=int, default=100)
parser.add_argument('--remove_frames', type=int, default=1500)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--adplr', action='store_true')
parser.add_argument('--jitc', action='store_true')
parser.add_argument('--downsample', type=int, default=0)
parser.add_argument('--msteps', type=int, default=4)
parser.add_argument('--resume', type=int, default=0)
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--gpus', type=int, default=0)
parser.add_argument('--rseed', type=int, default=42)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


import tensorflow as tf
print(tf.__version__)

gpus = tf.config.list_physical_devices('GPU') 

if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #log.info('{} Physical GPUs {} Logical GPUs'.format(len(gpus), len(logical_gpus)))



from phi.tf.flow import *
from phi.physics._boundaries import Domain, OPEN, STICKY as CLOSED
from phi import __version__
from tensorflow import keras
import random
import time
print("Phiflow version: {}".format(phi.__version__))


random.seed(args.rseed)
np.random.seed(args.rseed)
tf.random.set_seed(args.rseed)



# convert coordinates to phiflow objects

def convert_to_obj(geo_coor):
    
    #ar = geo_coor.values.numpy(('batch', 'x', 'y'))
    ar = geo_coor
    
    br_x1 = ar[0][0,1]
    br_x2 = ar[0][0,3]

    br_y1 = ar[0][0,4]
    br_y2 = ar[0][0,6]

    #print('base_rect', br_x1, br_x2, br_y1, br_y2)

    obj1 = (Box[br_x1:br_x2,br_y1:br_y2]) # base rectangle

    if (ar[0][1,0])==1: # 1 signifies a rectangle

        r_x1 = ar[0][1,1]
        r_x2 = ar[0][1,7]

        r_y1 = ar[0][1,4]
        r_y2 = ar[0][1,2]

        #print('rect 1', r_x1, r_x2, r_y1, r_y2)

        obj2 = (Box[r_x1:r_x2,r_y1:r_y2])

    elif (ar[0][1,0])==0: # 0 signifies a circle

        c_x1 = ar[0][1,1]
        c_y1 = ar[0][1,2]
        c_rad1 = ar[0][1,3]

        #print('circ 1', c_x1, c_y1, c_rad1)

        c_center = tensor((c_x1, c_y1), channel('vector'))

        obj2 = Sphere(center=c_center, radius=c_rad1)


    if (ar[0][2,0])==1: # 1 signifies a rectangle

        r_x1 = ar[0][2,1]
        r_x2 = ar[0][2,3]

        r_y1 = ar[0][2,4]
        r_y2 = ar[0][2,6]

        #print('rect 2', r_x1, r_x2, r_y1, r_y2)

        obj3 = (Box[r_x1:r_x2,r_y1:r_y2])

    elif (ar[0][2,0])==0: # 0 signifies a circle

        c_x1 = ar[0][2,1]
        c_y1 = ar[0][2,2]
        c_rad1 = ar[0][2,3]

        #print('circ 2', c_x1, c_y1, c_rad1)

        c_center = tensor((c_x1, c_y1), channel('vector'))

        obj3 = Sphere(center=c_center, radius=c_rad1)                


    if (ar[0][3,0])==1: # 1 signifies a rectangle

        r_x1 = ar[0][3,7]
        r_x2 = ar[0][3,1]

        r_y1 = ar[0][3,2]
        r_y2 = ar[0][3,4]

        #print('rect 3', r_x1, r_x2, r_y1, r_y2)

        obj4 = (Box[r_x1:r_x2,r_y1:r_y2])

    elif (ar[0][3,0])==0: # 0 signifies a circle

        c_x1 = ar[0][3,1]
        c_y1 = ar[0][3,2]
        c_rad1 = ar[0][3,3]

        #print('circ 3', c_x1, c_y1, c_rad1)

        c_center = tensor((c_x1, c_y1), channel('vector'))

        obj4 = Sphere(center=c_center, radius=c_rad1)          


    if (ar[0][4,0])==1: # 1 signifies a rectangle

        r_x1 = ar[0][4,3]
        r_x2 = ar[0][4,1]

        r_y1 = ar[0][4,6]
        r_y2 = ar[0][4,4]

        #print('rect 4', r_x1, r_x2, r_y1, r_y2)

        obj5 = (Box[r_x1:r_x2,r_y1:r_y2])

    elif (ar[0][4,0])==0: # 0 signifies a circle

        c_x1 = ar[0][4,1]
        c_y1 = ar[0][4,2]
        c_rad1 = ar[0][4,3]

        #print('circ 4', c_x1, c_y1, c_rad1)

        c_center = tensor((c_x1, c_y1), channel('vector'))

        obj5 = Sphere(center=c_center, radius=c_rad1)  


    if (ar[0][5,0])==1: # 1 signifies a rectangle

        r_x1 = ar[0][5,1]
        r_x2 = ar[0][5,3]

        r_y1 = ar[0][5,4]
        r_y2 = ar[0][5,6]

        #print('rect 5', r_x1, r_x2, r_y1, r_y2)

        obj6 = (Box[r_x1:r_x2,r_y1:r_y2])

    elif (ar[0][5,0])==0: # 0 signifies a circle

        c_x1 = ar[0][5,1]
        c_y1 = ar[0][5,2]
        c_rad1 = ar[0][5,3]

        #print('circ 5', c_x1, c_y1, c_rad1)

        c_center = tensor((c_x1, c_y1), channel('vector'))

        obj6 = Sphere(center=c_center, radius=c_rad1)  


    if (ar[0][6,0])==1: # 1 signifies a rectangle

        r_x1 = ar[0][6,1]
        r_x2 = ar[0][6,3]

        r_y1 = ar[0][6,4]
        r_y2 = ar[0][6,6]

        #print('rect 6', r_x1, r_x2, r_y1, r_y2)

        obj7 = (Box[r_x1:r_x2,r_y1:r_y2])

    elif (ar[0][6,0])==0: # 0 signifies a circle

        c_x1 = ar[0][6,1]
        c_y1 = ar[0][6,2]
        c_rad1 = ar[0][6,3]

        #print('circ 6', c_x1, c_y1, c_rad1)

        c_center = tensor((c_x1, c_y1), channel('vector'))

        obj7 = Sphere(center=c_center, radius=c_rad1) 


    multiple_obj = [obj1, obj2, obj3, obj4, obj5, obj6, obj7]

    obstacle = Obstacle(union(multiple_obj))
    
    return multiple_obj
    #return obj1, obj2, obj3, obj4, obj5, obj6, obj7


def downsample_field(data_u, data_v, data_ibm, downx):
    
    stacked = phi.math.stack([data_u,data_v], phi.math.channel('vector'))
    
    if downx==0:
        down_vel = phi.math.tensor(stacked)
        down_ibm = data_ibm
    
    if downx>0:
        for k in range(downx):

            down_vel = phi.math.downsample2x(stacked)
            down_ibm = phi.math.downsample2x(data_ibm)
            
            stacked = phi.math.stack([down_vel.vector[0],down_vel.vector[1]], phi.math.channel('vector'))
            data_ibm = down_ibm
    
    return down_vel.vector[0].numpy('x,y'), down_vel.vector[1].numpy('x,y'), down_ibm.numpy('x,y') 
    
    
total_frames = args.total_frames # total number of training frames from th avialable 3000 frames
sub_frames = 100
Nx = 768
Ny = 512
L = 24
H = 16

remove_transient = args.remove_frames 		# remove transient frames in the range 0-2900 (in multiple of 100, e.g., 0, 100, 200, ...2900)
end_frame = remove_transient + total_frames 	# window of frames to be considered

range_i = np.int32(remove_transient/sub_frames)
range_e = np.int32(end_frame/sub_frames)

downsample = args.downsample # 0 for no downsampling, 1 for downsampling by 2x, 2 for downsampling by 4x

configs = args.configs


def load_npz(file_name1,file_name2,range_i,range_e):

    alist_foam = []
    z=0
    for i in range(range_i,range_e,1):
                
        fname = file_name1 + 'comp_frames_range_' + str(i) + '.npz'
        data_of = np.load(fname)['arr_0']
        
        data_geo = np.loadtxt(file_name2, delimiter = ",", skiprows=0)
        
        for j in range(sub_frames):

            u_vel = np.transpose(np.reshape(data_of[j][:,0]*1.0, (Ny,Nx)))
            v_vel = np.transpose(np.reshape(data_of[j][:,1]*1.0, (Ny,Nx)))
            ibmask = np.transpose(np.reshape(data_of[j][:,3]*1.0, (Ny,Nx)))
            
            data_u = phi.math.tensor(u_vel,spatial('x','y'))
            data_v = phi.math.tensor(v_vel,spatial('x','y'))
            data_ibm = phi.math.tensor(ibmask,spatial('x','y'))

            down_u, down_v, down_ibm = downsample_field(data_u, data_v, data_ibm, downsample)

            re_u = down_u.reshape(1,down_u.shape[0], down_u.shape[1])
            re_v = down_v.reshape(1,down_v.shape[0], down_v.shape[1])
            re_ibm = down_ibm.reshape(1,down_ibm.shape[0], down_ibm.shape[1])
            re_coor = data_geo.reshape(1,data_geo.shape[0], data_geo.shape[1]) 
            #print(re_u.shape, re_ibm.shape)

            obj = convert_to_obj(re_coor)

            alist_foam.append([re_ibm,re_u,re_v,obj])
            arr = np.array(alist_foam,dtype=object)
    
    del data_of
    del alist_foam
    del data_geo
    
    return arr


start = time.time()

data_preloaded = {}

arr_foam = []
sim_name = []


for j in range (configs):

    suffix = '../../../../../media/brahmachary/data/Shuvayan_new_data/3_HR_mod_data/'
    
    path1 = suffix + 'foamExtend_folders/geo/geo_' + str(j) + '/flowfield_data/'
    path2 = suffix + '2_shape_coors/geo_' + str(j) + '.dat'
        
    arr_foam.append(load_npz(path1,path2,range_i,range_e)) 

    sim_name1 = 'openfoam_database/sim_' + str(j)
    sim_name.append(sim_name1)

for j in range (configs):
    
    data_preloaded[sim_name[j]] = arr_foam[j]
    #print(len(data_preloaded[sim_name[j]]))

end = time.time()

print('Time taken to load {} configs and {} training frames: {}'.format(configs, total_frames, end-start))


DT = 0.1

Nx = 128
Ny = 128

if downsample==1:
    Nx = int(Nx/2)
    Ny = int(Ny/2)
if downsample==2:
    Nx = int(Nx/4)
    Ny = int(Ny/4)
if downsample==3:
    Nx = int(Nx/8)
    Ny = int(Ny/8)
if downsample==4:
    Nx = int(Nx/16)
    Ny = int(Ny/16)    
    
offset = (L/Nx)*0.5

NU = 0.01
V = 1

class cyl_in_channel():
    def __init__(self, domain):
        self.domain = domain
    
        self.vel_BcMask = self.domain.staggered_grid(HardGeometryMask(Box[:offset, :]) )

    def step(self, mask_in, velocity_in, u_obj, seq, dt=DT):
                
        o_list = []
        
        for i in range(len(seq)):
            if seq[i]==1:
                o_list.append(Box[u_obj[i][0]:u_obj[i][1],u_obj[i][2]:u_obj[i][3]])

            if seq[i]==0:
                c_center = tensor((u_obj[i][0], u_obj[i][1]), channel('vector'))
                o_list.append(Sphere(center=c_center, radius=u_obj[i][2]))
                

        self.obstacle = [Obstacle(union(o_list))] 
        
        velocity = velocity_in
        pressure = None
        
        velocity = phi.flow.diffuse.explicit(velocity, NU, dt=dt)
        velocity = advect.mac_cormack(velocity, velocity, dt=dt)
        velocity = velocity*(1.0 - self.vel_BcMask) + self.vel_BcMask * (1,0)
        velocity, pressure = fluid.make_incompressible(velocity, self.obstacle, Solve('CG', 1e-5, 1e-5, 2500, x0=None))
        
        return [mask_in, velocity]


def network(inputs_dict):
    l_input = keras.layers.Input(**inputs_dict)
    block_0 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(l_input)
    block_0 = keras.layers.ReLU()(block_0)

    l_conv1 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(block_0)
    l_conv1 = keras.layers.ReLU()(l_conv1)
    l_conv2 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(l_conv1)
    l_skip1 = keras.layers.add([block_0, l_conv2])
    block_1 = keras.layers.ReLU()(l_skip1)

    l_conv3 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(block_1)
    l_conv3 = keras.layers.ReLU()(l_conv3)
    l_conv4 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(l_conv3)
    l_skip2 = keras.layers.add([block_1, l_conv4])
    block_2 = keras.layers.ReLU()(l_skip2)

    l_conv5 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(block_2)
    l_conv5 = keras.layers.ReLU()(l_conv5)
    l_conv6 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(l_conv5)
    l_skip3 = keras.layers.add([block_2, l_conv6])
    block_3 = keras.layers.ReLU()(l_skip3)

    l_conv7 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(block_3)
    l_conv7 = keras.layers.ReLU()(l_conv7)
    l_conv8 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(l_conv7)
    l_skip4 = keras.layers.add([block_3, l_conv8])
    block_4 = keras.layers.ReLU()(l_skip4)

    l_conv9 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(block_4)
    l_conv9 = keras.layers.ReLU()(l_conv9)
    l_conv10 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(l_conv9)
    l_skip5 = keras.layers.add([block_4, l_conv10])
    block_5 = keras.layers.ReLU()(l_skip5)
    
    l_conv11 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(block_5)
    l_conv11 = keras.layers.ReLU()(l_conv11)
    l_conv12 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(l_conv11)
    l_skip6 = keras.layers.add([block_5, l_conv12])
    block_6 = keras.layers.ReLU()(l_skip6)    
    
    l_conv13 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(block_6)
    l_conv13 = keras.layers.ReLU()(l_conv13)
    l_conv14 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(l_conv13)
    l_skip7 = keras.layers.add([block_6, l_conv14])
    block_7 = keras.layers.ReLU()(l_skip7)   
    
    l_conv15 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(block_7)
    l_conv15 = keras.layers.ReLU()(l_conv15)
    l_conv16 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(l_conv15)
    l_skip8 = keras.layers.add([block_7, l_conv16])
    block_8 = keras.layers.ReLU()(l_skip8) 
   
    l_conv17 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(block_8)
    l_conv17 = keras.layers.ReLU()(l_conv17)
    l_conv18 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(l_conv17)
    l_skip9 = keras.layers.add([block_8, l_conv18])
    block_9 = keras.layers.ReLU()(l_skip9) 
    
    l_conv19 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(block_9)
    l_conv19 = keras.layers.ReLU()(l_conv19)
    l_conv20 = keras.layers.Conv2D(filters=32, kernel_size=args.kernel_size, padding='same')(l_conv19)
    l_skip10 = keras.layers.add([block_9, l_conv20])
    block_10 = keras.layers.ReLU()(l_skip10)
    
    l_output = keras.layers.Conv2D(filters=2,  kernel_size=args.kernel_size, padding='same')(block_10)
    return keras.models.Model(inputs=l_input, outputs=l_output, name='long_resnet')        
        
    
def lr_schedule(epoch, current_lr):

    lr = current_lr
    if   epoch == 41: lr *= 0.5
    elif epoch == 31: lr *= 1e-1
    elif epoch == 21: lr *= 1e-1
    elif epoch == 11: lr *= 1e-1
    return lr


def to_keras(dens_vel_grid_array):
    
    x = math.stack(
        [
            dens_vel_grid_array[0].values,
            dens_vel_grid_array[1].vector['x'].x[:-1].values,
            dens_vel_grid_array[1].vector['y'].y[:-1].values,
        ],
        math.channel('channels')
    )
    return x

def to_phiflow(tf_tensor, domain):
     
    y = domain.staggered_grid(
        math.stack(
            [
                math.tensor(tf.pad(tf_tensor[..., 0], [(0,0), (0,1), (0,0)],"SYMMETRIC"), math.batch('batch'), math.spatial('x, y')), 
                math.tensor(tf.pad(tf_tensor[..., 1], [(0,0), (0,0), (0,1)],"SYMMETRIC"), math.batch('batch'), math.spatial('x, y'))      
            ],math.channel('vector')
        )
    )
    return y

class Dataset():
    def __init__(self, data_preloaded, num_frames, num_sims=None, batch_size=1, is_testset=False):
        self.epoch         = None
        self.epochIdx      = 0
        self.batch         = None
        self.batchIdx      = 0
        self.step          = None
        self.stepIdx       = 0

        self.dataPreloaded = data_preloaded
        self.batchSize     = batch_size

        self.numSims       = num_sims
        self.numBatches    = num_sims//batch_size
        self.numFrames     = num_frames
        self.numSteps      = num_frames
        
        if not is_testset:
            self.dataSims = ['openfoam_database/sim_%d'%i for i in range(num_sims) ]    
            
        else:
            self.dataSims = ['openfoam_database/sim_%d'%i for i in range(num_sims) ]

        self.dataFrames = [ np.arange(num_frames) for _ in self.dataSims ]  
        
        self.resolution = self.dataPreloaded[self.dataSims[0]][0][0].shape[1:4] 

        self.dataStats = {
            'std': (
                np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][0].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)*1) + 0, # mask
                np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][1].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)*1) + 0, # x-velocity
                np.std(np.concatenate([np.absolute(self.dataPreloaded[asim][i][2].reshape(-1)) for asim in self.dataSims for i in range(num_frames)], axis=-1)*1) + 0, # y-velocity
            )
        }

        if not is_testset:
            print("Data stats: "+format(self.dataStats))

    
    # re-shuffle data for next epoch
    def newEpoch(self, ep, exclude_tail=0, shuffle_data=True):
        self.numSteps = self.numFrames - exclude_tail
        simSteps = [ (asim, self.dataFrames[i][0:(len(self.dataFrames[i])-exclude_tail)]) for i,asim in enumerate(self.dataSims) ]
        
        sim_step_pair = []
        for i,_ in enumerate(simSteps):
            sim_step_pair += [ (i, astep) for astep in simSteps[i][1] ]  # (sim_idx, step) ...
            
        if shuffle_data: 
            random.shuffle(sim_step_pair)
        self.epoch = [ list(sim_step_pair[i*self.numSteps:(i+1)*self.numSteps]) for i in range(self.batchSize*self.numBatches) ]
        self.epochIdx += 1
        self.batchIdx = 0                    
        self.stepIdx = 0
     
        
    def nextBatch(self):  
        self.batchIdx += self.batchSize
        self.stepIdx = 0
        
    def nextStep(self):
        self.stepIdx += 1


# for class Dataset():
def getData(self, consecutive_frames):
                
    mask_hi = [
        np.concatenate([
            self.dataPreloaded[
                self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]] # sim_key
            ][
                self.epoch[self.batchIdx+i][self.stepIdx][1]+j # frames
            ][0] # variable
            for i in range(self.batchSize)
        ], axis=0) for j in range(consecutive_frames+1)
    ]
    u_hi = [
        np.concatenate([
            self.dataPreloaded[
                self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]] # sim_key
            ][
                self.epoch[self.batchIdx+i][self.stepIdx][1]+j # frames
            ][1] # variable
            for i in range(self.batchSize)
        ], axis=0) for j in range(consecutive_frames+1)
    ]
    v_hi = [
        np.concatenate([
            self.dataPreloaded[
                self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]] # sim_key
            ][
                self.epoch[self.batchIdx+i][self.stepIdx][1]+j # frames
            ][2] # variable
            for i in range(self.batchSize)
        ], axis=0) for j in range(consecutive_frames+1)
    ]
    
    coor_hi = [           self.dataPreloaded[
                self.dataSims[self.epoch[self.batchIdx+i][self.stepIdx][0]] # sim_key
            ][
                self.epoch[self.batchIdx+i][self.stepIdx][1] # frames
            ][3] # variable
            for i in range(self.batchSize)
        ]    

    return [mask_hi, u_hi, v_hi, coor_hi[0]]



nsims = configs # configs
batch_size = 1
simsteps = total_frames # frames

dataset = Dataset( data_preloaded=data_preloaded, num_frames=simsteps, num_sims=nsims, batch_size=batch_size)


msteps = args.msteps
source_res = list(dataset.resolution)

boundary_conditions = {
    'x':(phi.physics._boundaries.OPEN, phi.physics._boundaries.OPEN), 
    'y':(phi.physics._boundaries.OPEN, phi.physics._boundaries.OPEN) } 

domain = Domain(x=source_res[0], y=source_res[1], bounds=Box[0:L, 0:H], boundaries=boundary_conditions)

simulator = cyl_in_channel(domain=domain)

network = network(dict(shape=(source_res[0],source_res[1], 3)))
network.summary()



def training_step(mask_gt, vel_gt, obj, seq_l):
    with tf.GradientTape() as tape:
        
        prediction, correction = [ [mask_gt[0],vel_gt[0]]], [0] # ground truth as initial guess
  
        for i in range(msteps): # phiflow output
            
            prediction += [
                simulator.step(
                    mask_in=prediction[-1][0],
                    velocity_in=prediction[-1][1],
                    u_obj=obj,
                    seq=seq_l,
                )
            ]       

            model_input = to_keras(prediction[-1]) # staggered to centered
            model_input /= math.tensor([dataset.dataStats['std'][0], dataset.dataStats['std'][1], dataset.dataStats['std'][2]], channel('channels')) # [mask, u, v]
            model_out = network(model_input.native(['batch', 'x', 'y', 'channels']), training=True) #centered network output
            model_out *= [dataset.dataStats['std'][1], dataset.dataStats['std'][2]] # [u, v]

            correction += [ to_phiflow(model_out, domain)] # centered to staggered grid
            
            prediction[-1][1] = prediction[-1][1] + correction[-1] # correction to phiflow output
        
       
        loss_steps_x = [
            tf.nn.l2_loss(
                (
                    vel_gt[i].vector['x'].values.x[:].y[:].native(('batch', 'x', 'y'))
                    - prediction[i][1].vector['x'].values.x[:].y[:].native(('batch', 'x', 'y'))
                )/dataset.dataStats['std'][1]
            )
            for i in range(1,msteps+1)
        ]
        loss_steps_x_sum = tf.math.reduce_sum(loss_steps_x)
        
        loss_steps_y = [
            tf.nn.l2_loss(
                (
                    vel_gt[i].vector['y'].values.x[:].y[:].native(('batch', 'x', 'y'))
                    - prediction[i][1].vector['y'].values.x[:].y[:].native(('batch', 'x', 'y'))
                )/dataset.dataStats['std'][2]
            )
            for i in range(1,msteps+1)
        ]
        loss_steps_y_sum = tf.math.reduce_sum(loss_steps_y)


        loss = (loss_steps_x_sum + loss_steps_y_sum)/msteps

        gradients = tape.gradient(loss, network.trainable_variables)
        opt.apply_gradients(zip(gradients, network.trainable_variables))

        return math.tensor(loss)        


def obj_check(geom):
    
    if str(geom)=='Sphere()':
        cx = geom.center[0]
        cy = geom.center[1]
        crad = geom.radius
        dummy = 0
        obj_gt = math.tensor([cx,cy,crad,dummy])
        seq = 0
    else:
        obj_gt = math.tensor([geom.lower[0],geom.upper[0],geom.lower[1],geom.upper[1]])
        seq = 1
    return obj_gt, seq
    
    
training_step_jit = math.jit_compile(training_step)

current_lr = args.lr
EPOCHS = args.epochs

opt = tf.keras.optimizers.Adam(learning_rate=current_lr) 

resume = args.resume
if resume>0: 
    ld_network = keras.models.load_model('./nn_epoch{:04d}.h5'.format(resume)) 
    #ld_network = keras.models.load_model('./nn_final.h5') # or the last one
    network.set_weights(ld_network.get_weights())


save_dir = str(args.output) + '_' + str(args.dir_num)


print("Num GPUs Available: {}".format(len(tf.config.list_physical_devices('GPU'))))
print('Grid resolution: {}, downsample {}'.format(source_res, int(downsample)))
print('Remove Frames: {}'.format(remove_transient))
print('Test Frames: {} - {}'.format(remove_transient, remove_transient + total_frames))
print('Test Range: {} - {}'.format(range_i, range_e))
print('Test configs: {}'.format(configs))
print('Batch size: {}'.format(dataset.numBatches))
print('Training epochs: {}'.format(EPOCHS))
print('msteps: {}'.format(int(msteps)))
print('Random seed:  {}'.format(args.rseed))
print('Initial learning rate: {}'.format(current_lr))
print('Adaptive learning rate: {}'.format(args.adplr))
print('Jit compilation: {}'.format(args.jitc))
print('output dir:  {}\n'.format(save_dir))


Loss = []
Epochs = []


if os.path.exists(save_dir):
    print('{} folder exist !'.format(save_dir))
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
   
    with open(save_dir + "/output.txt", 'w') as f:
        f.write("GPUs: %d\n" %(len(tf.config.list_physical_devices('GPU'))))
        f.write('Grid resolution: %d x %d, downsample: %d\n' %(source_res[0],source_res[1], int(downsample)))
        f.write('Remove frames: %d\n' %remove_transient)
        f.write('Test range: %d - %d\n' %(remove_transient, remove_transient + total_frames))
        f.write('Test configs: %d - %d\n' %(range_i, range_e))
        f.write('Batch size: %d\n' %(dataset.numBatches))
        f.write('Training epochs: %d\n' %(EPOCHS))
        f.write('msteps: %d\n' %(msteps))
        f.write('Random seed: %d\n' %(args.rseed))
        f.write('Initial learning rate: %.8f\n' %(current_lr))
        f.write('Adaptive learning rate: %d\n' %(args.adplr))
        f.write('Jit compilation: %d\n' %(args.jitc))
        f.write('Output dir: %s\n' %(save_dir))
        
        f.close()
        
    with open(save_dir + "/network_summary.dat",'w') as f:
    	network.summary(print_fn=lambda x: f.write(x + '\n'))
        
     
 
steps = 0

if Epochs:
    xx = max(Epochs)
else:
    xx = 0

for j in range(EPOCHS):  # training
    dataset.newEpoch(j, exclude_tail=msteps)

    if j<resume:
        print('resume: skipping {} epoch'.format(j+1))
        steps += dataset.numSteps*dataset.numBatches
        continue
        
    current_lr = lr_schedule(j, current_lr) if args.adplr else args.lr

    for ib in range(dataset.numBatches):  # iterates through batches
        for i in range(dataset.numSteps): # iterates through simSteps keys

            batch = getData(dataset, consecutive_frames=msteps)

            mask_gt = [   
                domain.scalar_grid(
                    math.tensor(batch[0][k], math.batch('batch'), math.spatial('x, y'))
                ) for k in range(msteps+1)
            ]

            vel_gt = [
                domain.staggered_grid(
                    math.stack(
                        [
                            math.tensor(tf.pad(batch[1][k][...,:,:],[(0,0), (0,1), (0,0)],"SYMMETRIC"), math.batch('batch'), math.spatial('x, y')),
                            math.tensor(tf.pad(batch[2][k][...,:,:],[(0,0), (0,0), (0,1)],"SYMMETRIC"), math.batch('batch'), math.spatial('x, y')),
                        ], math.channel('vector')))

                for k in range(msteps+1)
            ]

            obj_1 = batch[3][0]
            obj_2 = batch[3][1]
            obj_3 = batch[3][2]
            obj_4 = batch[3][3]
            obj_5 = batch[3][4]
            obj_6 = batch[3][5]
            obj_7 = batch[3][6]
            
            obj_l = []
            seq_l = []
            for l in tf.range(len(batch[3])):
                obj_gt, seq = obj_check(batch[3][l])
                obj_l.append(obj_gt)
                seq_l.append(seq)
                
            if args.jitc:
            	loss = training_step_jit(mask_gt,vel_gt,obj_l,seq_l)
            else:
            	loss = training_step(mask_gt,vel_gt,obj_l,seq_l)
            	
            steps += 1
            print(' simStep {:04d}/{:04d}, batch {:03d}/{:03d}, epoch {:03d}/{:03d}, loss={}, lr={}'.format(i+1, dataset.numSteps, ib+1, dataset.numBatches, j+1, EPOCHS, loss,current_lr ))
            

            dataset.nextStep()

        dataset.nextBatch()

    network.save(save_dir + '/./nn_epoch{:04d}.h5'.format(j+1))

    Epochs.append(j+xx)
    Loss.append(loss)
    
    with open(save_dir + '/epoch_vs_loss.dat', 'ab') as f:
    	np.savetxt(f, [[j+xx,loss]], fmt="%.6f",delimiter=',')    


network.save(save_dir + '/./nn_final.h5');

print("Training done, saved NN")

l = np.asarray(Loss)
e = np.asarray(Epochs)

output = np.transpose([e+1, l])

np.savetxt(save_dir + '/final_loss.dat', output)


