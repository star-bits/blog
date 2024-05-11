# Terminal commands

## sh
```shell
# used space (du: disk usage, -s: summarize, -h: human-readable, *: all files and dirs)
du -sh *
# available space (df: disk free, -h: human-readable)
df -h

# items count (wc -l: 'word count' but for lines)
ls | wc -l
# list by largest first (-l: long format, -S: sort by size, -h: human-readable)
ls -lSh
# including hidden files (-a: all files including hidden files, -l long format)
ls -al

rm *.jpg

# monitor nvidia GPUs (-l 1: update interval set to 1 second)
nvidia-smi -l 1
# process status (-e: display all processes, -f: full-format including Process ID, CPU usage)
ps -ef
kill <pid> 

# enable shell
bash
# print current shell
echo $SHELL
# restart shell
source ~/.bashrc
# or, because `.` is an alias for `source` in bash and zsh,
. ~/.bashrc

# web get
wget <url>

# tape archive (-c: create, -t: list contents, -x: extract, -v: berbose, -f specifies file name)
tar cvf A.tar A
tar tvf A.tar
tar xvf A.tar

# zip (-r: recursively, -l: list)
zip -r A.zip A
unzip -l A.zip
unzip A.zip

# history
history
!<history_number>

# global regular expression print
ls | grep "a"
history | grep "a"

# private IP address
ipconfig getifaddr en0
# public IP address
curl ifconfig.me
```

## vi
```shell
vi file_name.txt

# insert mode
i
# command mode
(Press the 'esc' key)
# last line mode
:
# quit (from the last line mode)
q
# write and quit (from the last line mode)
wq
```

## brew
```shell
brew install miniconda
brew install scrcpy
brew install rename
```

## conda
```shell
conda create --name xcda python=3.8

conda activate xcda
conda deactivate

conda config --set auto_activate_base false

conda env list

# packages can be installed from Anaconda, conda-forge, or PyPI

# `requirements.txt` equivalent
conda env export -f environment.yml 
conda env create -n xcda -f /path/to/environment.yml
conda env update -n xcda -f /path/to/environment.yml

conda env remove -n xcda
```

## venv
```shell
python3 -m venv xvnv

. xvnv/bin/activate
deactivate

pip freeze
```

## virtualenv
```shell
virtualenv --python=python3 xvlv

. xvlv/bin/activate
deactivate

pip freeze
```

## Default Python interpreter
```shell
type -a python3
python3 is /opt/homebrew/bin/python3
python3 is /usr/bin/python3

/opt/homebrew/bin/python3 --version

brew unlink python
brew search python@
brew install python@3.10

/opt/homebrew/Cellar/python@3.10
```

## tmux
```shell
apt-get install tmux

tmux new -s <session_name>
tmux ls
tmux attach -t <session_name>
tmux kill-session -t <session_name>
```

## git
https://git-scm.com/download/win
```shell
git init # ⭐
# to add changes to the staging area:
git add <file_name>
git add . # ⭐
# to create a new commit with the changes in the staging area:
git commit -m "commit message" # ⭐

# to create `.gitignore`:
touch .gitignore # ⭐
# to apply changes to the `.gitignore` on files that are already being tracked:
git rm -r --cached . # ⭐
# to add a folder named xvnv and all its contents to `.gitignore`:
echo 'xvnv/' > .gitignore # ⭐

# HEAD: where any new commits you make will be based on.
# master: main branch of your repository.
# detached HEAD state: when HEAD is not pointing to the latest commit on a branch.
# checkout: command to switch between different branches or commits.
git checkout <commit_hash>
git checkout <tag>
# checkout to a branch moves HEAD to point to the latest commit on that branch.
git checkout <branch_name>
git checkout master
# to show a list of all branches:
git branch
# to create a branch:
git branch <branch_name>
# switch is equivalent to checkout, but works between branches only.
git switch <branch_name>
git merge <branch_name>

# origin: where remote repository is synced at.
# to add a remote repository and give it a name 'origin':
git remote add origin <repository_url> # ⭐
git remote show origin
git remote remove origin
# to show a list of remote repositories:
git remote -v

git fetch origin
git merge origin/<remote_branch_name>
# fetch + merge = pull
git pull origin <remote_branch_name>
git push origin <remote_branch_name> # ⭐
# to push only with `git push` without specifying remote branch:
git push --set-upstream origin <remote_branch_name> # ⭐

# to find a commit hash to go back:
git reflog
git reflog <branch_name>
# reset moves the current branch to a previous commit.
git reset --hard <commit_hash>
# unlike reset, revert creates a new commit instead of modifying the commit history.
# revert creates a new commit that undoes the changes introduced by that commit.
git revert HEAD
git revert <commit_hash>

git log --oneline --all --graph
git status

git config --global user.name
git config --global user.name "star-bits"
git config --global user.email
git config --global user.email "star-bits@outlook.com"
```

## docker
```shell
# build an image using the `Dockerfile` in the current directory, tag it as "my-image"
docker build -t my-image .

# run a container from the "my-image" image in detached mode, map port 8000 in the container to port 8000 on the host
docker run -d -p 8000:8000 --name my-container my-image
# run a container with a mounted volume, allowing changes to the source code on the host to be immediately reflected inside the container
docker run -d -p 8000:8000 -v $(pwd):/wd --name my-container my-image

# start a new bash session inside the running "my-container"
docker exec -it my-container /bin/bash

# stop the running "my-container"
docker stop my-container

# remove the "my-container"
docker rm my-container

# remove all stopped containers, all networks not used by at least one container, all dangling images, and all build cache
docker system prune
```

### `Dockerfile` example
```shell
# official python runtime as a parent image
FROM python:3.8-slim

# set the working directory in the container to /app
WORKDIR /wd

# add the current directory contents into the container at /app
ADD . /wd

# install packages
RUN pip install --no-cache-dir -r requirements.txt

# make port 80 available to the world outside this container
EXPOSE 80

# run `hello.py` when the container launches
CMD ["python", "hello.py"]
```
