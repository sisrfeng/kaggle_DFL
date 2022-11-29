This folder mimics "/kaggle"  in kaggle' s kernel.   
(The repo's name was `s_kaggle`, and `s` in `s_kaggle` means "slash" ) 

# Usage


1. `cd` to this repo.
2. `mkdir work`
3. May need to modify some dataset related paths in pre_proc.py and run_train.zsh. 
4. `cd ~ &&  git clone https://gitee.com/llwwff/timm__Torch-Image-Models &&  cd timm__Torch-Image-Models`  
5. `ln <Your_timm_Path>/timm__Torch-Image-Models <Your_competition_repo_path>/upload/timm`                 
6. Modify some alias in run_train.zsh, (see comments there) 
7. `cd working  &&  python pre_proc.py && bash run_train.zsh`




