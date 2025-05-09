# gUnet_sheet
#Total of 3 lines of code to be edited in order to use the code. Details shown below.


##After uploading dataset input, set dataset variables inside the code like this:

args.train_set = '/kaggle/input/haze4k-t'  # training dataset path
args.val_set = '/kaggle/input/haze4k-v'    # validation dataset path

##Go to kaggle input section --> click on upload --> go to new model --> upload the gunet_t.pth file --> put desired model name --> select pytprch as framework --> Select license as MIT -->upload model --> select and copy newly uploaded model folder path and put in place of the value of the variable args.load_dir inside the code as shown below.

args.load_dir= '/kaggle/input/gunet_t/pytorch/default/1/'

