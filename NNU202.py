"""
author:zhangjie
data :20201021
School of Computer and Electronic Information, Nanjing Normal University
I've integrated some common methods for working with data
"""
import skvideo.io
import os
from skimage.transform import resize
from skimage.io import imsave,imread
import cv2
import glob as gb
import numpy as np
import h5py
from tqdm import tqdm
import logging


video_root_path = "./video" #文件夹自己设定
frame_root_path = "./frames" #生成成功的frame都放在这个文件夹中
size1 = 224  #尺寸自己设定
size2 = 224
def video2frame(dataset,train_or_test,size1,size2):
    """
    :param dataset: your dataset
    :param train_or_test: choose which file
    :param size1: setting your frame length
    :param size2: setting your frame width
    :return: video to frame
    It is recommended to install FFMPEG software
    """
    video_path = os.path.join(video_root_path,dataset,'{}_videos'.format(train_or_test))
    frame_path = os.path.join(frame_root_path,dataset,'{}_frames'.format(train_or_test))
    os.makedirs(frame_path,exist_ok=True)

    for video_file in os.listdir(video_path):#这里是遍历文件，不是遍历文件数量，文件数量是range(len(yourfile))
        if video_file.lower().endswith(('.avi','.mp4')):
            print('==>'+os.path.join(video_path,video_file))
            vid_frame_path = os.path.join(frame_path, os.path.basename(video_file).split('.')[0])#就是生成每个帧的文件夹目录
            """
            #os.path.basename返回path最后的文件名。如果path以/或\结尾，那么就会返回空值。即os.path.split(path)的第二个元素。
            比如:path = 'D:/MYroot'   os.path.basename(path)=MYroot
            或者：video_file = './video/01.avi'
            print(os.path.basename(video_file).split('.')[0])----->01
            """
            os.makedirs(vid_frame_path,exist_ok=True)
            vidcap = skvideo.io.vreader(os.path.join(video_path,video_file))
            count = 1
            for image in vidcap:
                image = resize(image,(size1,size2),mode='reflect')
                imsave(os.path.join(vid_frame_path,'{:05d}.jpg'.format(count)),image) #生成图片 五位数
                count += 1

#def rename():
    #video = os.path

# def resize():
    pass

def frame2video(dataset,train_or_test,size1,size2):
    """
    :return: 将帧回到视频
    """
    video_path = os.path.join(video_root_path, dataset, '{}_videoss'.format(train_or_test)) #生成视频的路径
    frame_path = os.path.join(frame_root_path,dataset,'{}_frames'.format(train_or_test))  #原视频帧的路径
    os.makedirs(video_path,exist_ok=True) #创建生成视频的路径

    for file_number in range(len(os.listdir(frame_path))):
        file_number = file_number+1 #文件是从01开始，所以需要要 +1
        file_number = str(file_number)
        file_number = file_number.zfill(2)

        img_path = os.path.join(frame_path,'{}'.format(file_number),'*.jpg') #*.jpg是帧后缀，匹配所有图片，可以根据自己的文件后缀去修改
        img_path = gb.glob(img_path)
        video_path = os.path.join('{}/{}/{}_videoss//'.format(video_root_path,dataset,train_or_test),'{}.avi'.format(file_number)) #这句有问题
        videoWriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 25, (size1, size2))

        for path in img_path:
            img = cv2.imread(path)
            img = cv2.resize(img, (size1, size2))
            videoWriter.write(img)

def FD():
    """
    :return: 邻帧差分算法
    """
    pass

def TFD():
    """
    :return: 三帧差分算法
    """
    pass


def foldergen():
    """
    :return: 批量生成文件夹
    """
    pass

'''接下来几个函数的功能是frames-->meanframe.npy-->train/test_frame_{1,2...}.npy-->dataset_{1,2...}.h5(每个视频生成一个h5)-->dataset_train_t.h5(总数据集的h5文件)'''
def calc_mean(dataset,train_or_test,size1,size2):
    """
    :param dataset: avenue/ped1/ped2/shanghai
    :param framepath: your path
    :param size1:your dataset size length(such as 640*360? or 227*227? or 224*224)  size大小要和frame的大小对应
    :param size2：width
    :return:mean_frame_size.npy
    """
    frame_path = os.path.join(frame_root_path,dataset,'{}_frames'.format(train_or_test))
    count = 0
    frame_sum = np.zeros((size1,size2)).astype('float64')

    for frame_folder in os.listdir(frame_path):
        print('==>'+os.path.join(frame_path,frame_folder))
        for frame_file in os.listdir(os.path.join(frame_path,frame_folder)):
            frame_filename = os.path.join(frame_path,frame_folder,frame_file)
            frame_value = imread(frame_filename,as_gray=True)#若报错就改成as_grey,cv2版本问题
            assert (0. <= frame_value.all() <= 1.)
            frame_sum += frame_value
            count += 1

        frame_mean = frame_sum/count
        assert (0. <= frame_mean.all() <= 1.)
        np.save(os.path.join(frame_root_path,dataset,'{0}_mean_frame_{1}.npy'.format(train_or_test,size1)),frame_mean)

def substract_mean(dataset,train_or_test,size1,size2):
    frame_mean = np.load(os.path.join(frame_root_path,dataset,'{0}_mean_frame_{1}.npy'.format(train_or_test,size1)))
    frame_path = os.path.join(frame_root_path,dataset,'{}_frames'.format(train_or_test))
    for frame_folder in os.listdir(frame_path):
        print('==>'+os.path.join(frame_path,frame_folder))
        T_frames = '{}_frames'.format(train_or_test)
        T_frames_vid = []
        for frame_file in sorted(os.listdir(os.path.join(frame_path,frame_folder))):
            frame_filename = os.path.join(frame_path,frame_folder,frame_file)
            frame_value = imread(frame_filename,as_gray = True)#若报错就改成as_grey,cv2版本问题
            assert (0. <= frame_value.all()<= 1.)
            frame_value -= frame_mean #每个视频的帧都减去平均值
            T_frames_vid.append(frame_value)
        T_frames_vid = np.array(T_frames_vid)
        np.save(os.path.join(frame_root_path,dataset,'{0}_frames_{1}.npy'.format(train_or_test,frame_folder)),T_frames_vid)

def build_h5(dataset,train_or_test,t,size1,size2):
    print('==>{} {}'.format(dataset,train_or_test))
    def build_volume(train_or_test, num_videos, time_length):
        #num_videos = len(os.listdir(os.path.join(frame_root_path, '{}/{}_frames'.format('ped1', 'train'))))   #测试用
        for i in tqdm(range(num_videos)):  # tqdm输出一个类似进度条的东西 100%|██████████| 21/21 [00:00<?, ?it/s]
            data_frames = np.load(os.path.join(frame_root_path, '{}/{}_frames_{:02d}.npy'.format(dataset, train_or_test,i + 1)))  # 加载每一个train/test的npy文件
            data_frames = np.expand_dims(data_frames, axis=-1)  # 增加一维
            num_frames = data_frames.shape[0]  # 一共有多少帧

            data_only_frames = np.zeros((num_frames-time_length,time_length,size1,size2,1)).astype('float16')
            vol = 0
            for j in range(num_frames-time_length):
                data_only_frames[vol] = data_frames[j:j+time_length]#Read a single volume  time_length是10 所以是每次读取十张 先是1到10 然后是2到11 然后是3到12这样
                vol += 1

            with h5py.File(os.path.join(frame_root_path,'{0}/{1}_h5_t{2}/{0}_{3:02d}.h5'.format(dataset,train_or_test,time_length,i+1)),'w') as f:#创建训练集和测试集的h5文件
                if train_or_test == 'train':
                    np.random.shuffle(data_only_frames) #Scramble training set
                f['data'] = data_only_frames
    os.makedirs(os.path.join(frame_root_path,'{}/{}_h5_t{}'.format(dataset,train_or_test,t)),exist_ok=True)
    num_videos = len(os.listdir(os.path.join(frame_root_path,'{}/{}_frames'.format(dataset,train_or_test))))
    build_volume(train_or_test,num_videos,time_length=t)

def combine_dataset(dataset,t,size1,size2):
    print('==> {}'.format(dataset))
    output_file = h5py.File(os.path.join(frame_root_path,'{0}/{0}_train_t{1}.h5'.format(dataset,t)),'w')
    h5_folder = os.path.join(frame_root_path,'{}/train_h5_t{}'.format(dataset,t))
    filelist = sorted([os.path.join(h5_folder,item) for item in os.listdir(h5_folder)])
    #print(filelist)#for test 输出：['./frames\\ped1/train_h5_t10\\ped1_01.h5', './frames\\ped1/train_h5_t10\\ped1_02.h5']
    """keep track of the total number of rows"""
    total_rows = 0
    for n, f in enumerate(tqdm(filelist)):
        your_data_file = h5py.File(f,'r')
        your_data = your_data_file['data']
        total_rows = total_rows + your_data.shape[0]

        if n == 0:
            """first file: create the dummy dataset with no max shape"""
            create_dataset = output_file.create_dataset('data', (total_rows, t, size1, size2, 1),
                                                        maxshape=(None, t, size1, size2, 1))
            """fill the first section of the dataset"""
            create_dataset[:, :] = your_data
            where_to_start_appending = total_rows

        else:
            # resize the dataset to accomodate the new data
            create_dataset.resize(total_rows, axis=0)
            create_dataset[where_to_start_appending:total_rows, :] = your_data
            where_to_start_appending = total_rows

    output_file.close()

logger = logging.getLogger() #用于下面的logger
def preprocess_data(logger,dataset,t):
    '''第一步生成mean frames'''
    # Step 1: Calculate the mean frame of all training frames
    # Check if mean frame file exists for the dataset
    # If the file exists, then we can skip re-generating the file
    # Else calculate and generate mean file
    logger.debug('Step  1/4:Check if mean frame exists for {}'.format(dataset))
    mean_frame_file = os.path.join(frame_root_path,dataset,'train_mean_frame_{}.npy'.format(size1))
    train_frame_path = os.path.join(frame_root_path,dataset,'train_frames')
    test_frame_path = os.path.join(frame_root_path,dataset,'test_frames')
    if not os.path.isfile(mean_frame_file):
        '''The frames must have already been extracted from training and testing videos'''
        assert (os.path.isdir(train_frame_path))
        assert (os.path.isdir(test_frame_path))
        logger.info("Step 1/4: Calculating mean frame for {}".format(dataset))
        calc_mean(dataset,'train',size1,size2)
        calc_mean(dataset,'test',size1,size2)

        '''第二步生成train/test_frames_{}.npy'''
        # Step 2: Subtract mean frame from each training and testing frames
        # Check if training & testing frames are already been subtracted
        # If the file exists, then we can skip re-generating the file
        logger.debug('Step 2/4: Check if training/testing_frames_videoID.npy exists for {}".format(dataset))')
        try:
            for frame_folder in os.listdir(train_frame_path):
                train_frame_npy = os.path.join(frame_root_path,dataset,'train_frames_{}.npy'.format(frame_folder))
                assert (os.path.isfile(train_frame_npy))
            for frame_folder in os.listdir(test_frame_path):
                test_frame_npy = os.path.join(frame_root_path,dataset,'test_frames_{}.npy'.format(frame_folder))
                assert (os.path.isfile(test_frame_npy))

        except AssertionError:
            '''if all or some frames have not been substracted, then generate those file'''
            logger.info("Step 2/4: Subtracting mean frame for {}".format(dataset))
            substract_mean(dataset,'train',size1,size2)
            substract_mean(dataset,'test',size1,size2)

    '''第三步：生成ped1_train/test.h5'''
    # Step 3: Generate small video volumes from the mean-subtracted frames and dump into h5 files (grouped by video ID)
    # Check if those h5 files have already been generated
    # If the file exists, then skip this step
    logger.debug("Step 3/4: Check if individual h5 files exists for {}".format(dataset))
    for train_or_test in ('train','test'):
        try:
            h5_folder = os.path.join(frame_root_path,'{}/{}_h5_t{}'.format(dataset,train_or_test,t))
            assert (os.path.isfile(h5_folder))
            num_videos = len(os.listdir(os.path.join(frame_root_path,'{}/{}_frames'.format(dataset,train_or_test))))
            for i in range(num_videos):
                h5_file = os.path.join(frame_root_path,'{0}/{1}_h5_{2}/{0}_{3:02d}.h5'.format(dataset,train_or_test,t,i+1))
                assert (os.path.isfile(h5_file))
        except AssertionError:
            logger.info("Step 3/4: Generating volumes for {} {} set".format(dataset, train_or_test))
            build_h5(dataset,'train',t,size1,size2)
            build_h5(dataset,'test',t,size1,size2)

    '''第四步：生成ped1_train_t10.h5(举例)'''
    # Step 4: Combine small h5 files into one big h5 file
    # Check if this big h5 file is already been generated
    # If the file exists, then skip this step
    logger.debug("Step 4/4: Check if individual h5 files have already been combined for {}".format(dataset))
    train_h5 = os.path.join(frame_root_path,'{0}/{0}_train_t{1}.h5'.format(dataset,t))
    if not os.path.isfile(train_h5):
        logger.info("Step 4/4: Combining h5 files for {}".format(dataset))
        combine_dataset(dataset,t,size1,size2)

    logger.info('Preprocessing is completed~')






"""here are for test"""
#video2frame('ped1','train',224,224) #成功
#calc_mean('ped1','train',224,224) #success
#substract_mean('ped1','train',224,224) #success
#build_h5('ped1','train',10,224,224) #success
#combine_dataset('ped1',10,224,224) #success
#preprocess_data(logger,'ped1',10)
#frame2video('ped1','test',size1,size2)
