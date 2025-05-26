# Graphic cards
gc = "1"

# Model
mode = 'train'
batch_size = 32
img_size = (512,512)
input_shape = (512,512,3)
num_classes = 4
epochs = 25
model = "resnet"
lr = 1e-04

# saved_model
s_dir ='save/'
saved_model = {'vlc': s_dir + 'vlc_best.pth',
               'grx_mv': s_dir + 'grx_mv_best.pth',
               'grx_cr': s_dir + 'grx_cr_best.pth',
               'vlc_fine_grx_mv': s_dir + 'vlc_fine_grx_mv_best.pth',
               'vlc_fine_grx_cr': s_dir + 'vlc_fine_grx_cr_best.pth',
               'both_mv': s_dir + 'both_mv_best.pth',
               'both_cr': s_dir + 'both_cr_best.pth'}
