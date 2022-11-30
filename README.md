This folder mimics "/kaggle"  in kaggle kernel.   
(The repo's name in gitee is `s_kaggle`, and `s` in `s_kaggle` means "slash" ) 

# Usage 

1. `cd` to this repo
2. `mkdir work`
3. May need to modify some dataset related paths in these files under working/ : 
    pre_proc__val_sec_as_time.py, pre_proc__train_frm_as_time, run_train.zsh.
    `cd working && python pre_proc__val_sec_as_time.py  &&  python pre_proc__train_frm_as_time.py`

4. `git clone https://gitee.com/llwwff/timm__Torch-Image-Models &&  cd timm__Torch-Image-Models` #  then you can modify the timm repo and do not need to pip install timm. (So far, my fork of timm has no functional modifications. I've just added some comments and align some codes)  
5. `ln -s timm__Torch-Image-Models <Where_this_repo_is>/upload/timm`                 
6. Modify some alias in run_train.zsh (see comments there) 
7. `cd <Where_this_repo_is>/working  &&  python pre_proc.py && bash run_train.zsh`    
